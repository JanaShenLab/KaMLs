import torch
import os.path as osp
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse, to_networkx
from torch_geometric.data import Data, Batch, DataListLoader
from torch_geometric.loader import DataLoader
import h5py
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import networkx as nx
from torch.nn import Linear, L1Loss
from torch_geometric.nn import GATConv
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros, uniform
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.BatchNorm = nn.BatchNorm1d(num_features=42, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
    ):
        # forward_type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # forward_type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #torch.manual_seed(1)
        
        self.conv1 = GATConv(42, 42) ### changed the number of node_feat based on input
        self.conv2 = GATConv(42, 42)
        self.conv3 = GATConv(42, 42)
        
        self.fc1 = torch.nn.Linear(42, 32)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index, edge_attr, batch):
                
        #act = torch.nn.Tanhshrink()
        act = F.relu
        dropout = torch.nn.Dropout(p = 0.2)
        
        h = self.conv1(x, edge_index, edge_attr)
        h = h.tanh()
        h = self.conv2(h, edge_index, edge_attr)
        h = h.tanh()
        h = self.conv3(h, edge_index, edge_attr)
        h = h.tanh()
        h = global_mean_pool(h, batch)
        
        h = act(self.bn1(self.fc1(h)))
        h = dropout(h)
        h = act(self.bn2(self.fc2(h)))
        h = dropout(h)
        h = self.fc3(h)
        
        out = h
        
        return out

model = GAT()

### load the pre-trained weight based on PHMD549
state_dict = torch.load('/home/rotation/mingzhe/his_pka/PHMD549/dropp2p2_lr0005/layer_v3/REG_GAT_exptFv32_sphere_sgkf_selftest_graphein_arl_E0f0.pth')
with torch.no_grad():
    model.conv1.att_src.copy_(state_dict['conv1.att_src'])
    model.conv1.att_dst.copy_(state_dict['conv1.att_dst'])
    model.conv1.bias.copy_(state_dict['conv1.bias'])
    model.conv1.BatchNorm.weight.copy_(state_dict['conv1.BatchNorm.weight'])
    model.conv1.BatchNorm.bias.copy_(state_dict['conv1.BatchNorm.bias'])
    model.conv1.BatchNorm.running_mean.copy_(state_dict['conv1.BatchNorm.running_mean'])
    model.conv1.BatchNorm.running_var.copy_(state_dict['conv1.BatchNorm.running_var'])
    model.conv1.BatchNorm.num_batches_tracked.copy_(state_dict['conv1.BatchNorm.num_batches_tracked'])
    model.conv1.lin.weight.copy_(state_dict['conv1.lin.weight'])
    model.conv2.att_src.copy_(state_dict['conv2.att_src'])
    model.conv2.att_dst.copy_(state_dict['conv2.att_dst'])
    model.conv2.bias.copy_(state_dict['conv2.bias'])
    model.conv2.BatchNorm.weight.copy_(state_dict['conv2.BatchNorm.weight'])
    model.conv2.BatchNorm.bias.copy_(state_dict['conv2.BatchNorm.bias'])
    model.conv2.BatchNorm.running_mean.copy_(state_dict['conv2.BatchNorm.running_mean'])
    model.conv2.BatchNorm.running_var.copy_(state_dict['conv2.BatchNorm.running_var'])
    model.conv2.BatchNorm.num_batches_tracked.copy_(state_dict['conv2.BatchNorm.num_batches_tracked'])
    model.conv2.lin.weight.copy_(state_dict['conv2.lin.weight'])
    model.conv3.att_src.copy_(state_dict['conv3.att_src'])
    model.conv3.att_dst.copy_(state_dict['conv3.att_dst'])
    model.conv3.bias.copy_(state_dict['conv3.bias'])
    model.conv3.BatchNorm.weight.copy_(state_dict['conv3.BatchNorm.weight'])
    model.conv3.BatchNorm.bias.copy_(state_dict['conv3.BatchNorm.bias'])
    model.conv3.BatchNorm.running_mean.copy_(state_dict['conv3.BatchNorm.running_mean'])
    model.conv3.BatchNorm.running_var.copy_(state_dict['conv3.BatchNorm.running_var'])
    model.conv3.BatchNorm.num_batches_tracked.copy_(state_dict['conv3.BatchNorm.num_batches_tracked'])
    model.conv3.lin.weight.copy_(state_dict['conv3.lin.weight'])
    model.fc1.weight.copy_(state_dict['fc1.weight'])
    model.fc1.bias.copy_(state_dict['fc1.bias'])
    model.fc2.weight.copy_(state_dict['fc2.weight'])
    model.fc2.bias.copy_(state_dict['fc2.bias'])
    model.fc3.weight.copy_(state_dict['fc3.weight'])
    model.fc3.bias.copy_(state_dict['fc3.bias'])

# freeze the GAT layer weights
#for param in model.conv1.parameters():
#    param.requires_grad = False
#for param in model.conv2.parameters():
#    param.requires_grad = False
#for param in model.conv3.parameters():
#    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

def train(r_tl, train_res_li, train_pred_li, train_expt_li):
    model.train()
    
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        train_loss = criterion(out, data.y)
        train_r2 = r2_score(data.y.detach().numpy(), out.detach().numpy())
        r_tl += train_loss.item()*len(data.resname)
        res = data.resname
        pka = data.y.numpy()
        OUT = out.detach().numpy()
        train_res_li.extend(res)
        train_pred_li.extend(OUT)
        train_expt_li.extend(pka)
        #train_r = torch.corrcoef(torch.tensor([data.y.detach().numpy(), out.detach().numpy()]))
        #loss = criterion(out.squeeze(), data.y.squeeze())
        train_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    r_tl /= len(train_dataset_list)
    r_tl = r_tl ** 0.5
    return train_loss, train_r2, r_tl, train_res_li, train_pred_li, train_expt_li

def val(loader, r_vl, val_res_li, val_pred_li, val_expt_li):
    model.eval()

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        res = data.resname
        pka = data.y.numpy()
        OUT = out.detach().numpy()
        val_res_li.extend(res)
        val_pred_li.extend(OUT)
        val_expt_li.extend(pka)
        #mae = torch.mean(torch.abs(torch.sub(out, data.y)))
        loss = criterion(out, data.y)
        r2 = r2_score(data.y.detach().numpy(), out.detach().numpy())
        r_vl += loss.item()*len(data.resname)
        #r = torch.corrcoef([data.y.detach().numpy(), out.detach().numpy()])
        #r = torch.corrcoef(torch.tensor([data.y.detach().numpy(), out.detach().numpy()]))
    r_vl /= len(val_dataset_list)
    r_vl = r_vl ** 0.5
    return loss, r2, r_vl, val_res_li, val_pred_li, val_expt_li

def test(loader):
    model.eval()

    for data in loader:  # Iterate in batches over the validation dataset.
        pred_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pka = data.y
        res = data.resname
        #mae = torch.mean(torch.abs(torch.sub(pred_out, data.y)))
        #loss = torch.sqrt(criterion(out, data.y))
        #r2 = r2_score(data.y.detach().numpy(), pred_out.detach().numpy())
    return pred_out, pka, res
    #return pred_out

class EarlyStopper:
    def __init__(self, patience = 50, min_delta = 0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, train_loss, val_loss, train_res_li, train_pred_li, train_expt_li, val_res_li, val_pred_li, val_expt_li):
        if val_loss < self.min_val_loss:
            if abs(train_loss - val_loss) <= 0.35:
                #if train_r2 > 0.3:
                self.min_val_loss = val_loss
                torch.save(model.state_dict(), f"{pth}/{model_name}/v8/dropp2p2_lr0005/layer_v3_pretrain/PHMD549/dpka_res_split_v3_10_freeze0/{model_name}_E{n_expt}f{fold}.pth")
                pred_out, pka, res = test(test_loader)
                df_pred = pd.DataFrame(res)
                df_pred['expt'] = pka.numpy()
                df_pred['pred'] = pred_out.detach().numpy()
                df_pred.to_csv(f'{pth}/{model_name}/v8/dropp2p2_lr0005/layer_v3_pretrain/PHMD549/dpka_res_split_v3_10_freeze0/{model_name}_E{n_expt}f{fold}_predictions.csv', index = False)
                df_record.loc[len(df_record.index)] = [epoch, epoch, epoch, epoch, epoch]
                df_train_detail = pd.DataFrame(train_res_li)
                df_train_detail['expt'] = np.concatenate(train_expt_li)
                df_train_detail['pred'] = np.concatenate(train_pred_li)
                df_train_detail.to_csv(f'{pth}/{model_name}/v8/dropp2p2_lr0005/layer_v3_pretrain/PHMD549/dpka_res_split_v3_10_freeze0/{model_name}_E{n_expt}f{fold}_traindtl.csv', index = False)
                df_val_detail = pd.DataFrame(val_res_li)
                df_val_detail['expt'] = np.concatenate(val_expt_li)
                df_val_detail['pred'] = np.concatenate(val_pred_li)
                df_val_detail.to_csv(f'{pth}/{model_name}/v8/dropp2p2_lr0005/layer_v3_pretrain/PHMD549/dpka_res_split_v3_10_freeze0/{model_name}_E{n_expt}f{fold}_valdtl.csv', index = False)
                self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

pth = "/home/rotation/mingzhe/his_pka/gnn_reg/sphere_20"
model_name = 'REG_GAT_exptFv32_sphere_sgkf_selftest_graphein_arl'
batch_size = 64
data_pth = "/home/rotation/mingzhe/his_pka/split_restype_v3"

n_aug_in_test = 0
for n_expt in [%n_expt%]:
    for fold in [%n_fold%]:
        df_test = pd.read_csv(f'{data_pth}/expt{n_expt}_test.csv')
        df_val = pd.read_csv(f'{data_pth}/expt{n_expt}f{fold}_validation_10.csv')
        df_train = pd.read_csv(f'{data_pth}/expt{n_expt}f{fold}_train_10.csv')
        train_dataset_list = []
        test_dataset_list = []
        val_dataset_list = []
        with h5py.File(f"/home/rotation/mingzhe/his_pka/regression/split%n_expt%_restype_v3_stded_dpka.hdf", "r") as f:
            for name in list(f):
                if name in ['1STY-A-124']:
                    pass
                else:
                    data = torch.load(f'{model_name}/feat_v8_dpka_split%n_expt%_restype_v3/{name}.pt')
                    if name in df_test['info'].values:
                        test_dataset_list.append(data)
                    if name in df_val['info'].values:
                        val_dataset_list.append(data)
                    if name in df_train['info'].values:
                        train_dataset_list.append(data)
            print(len(train_dataset_list), len(val_dataset_list))
            train_loader = DataLoader(train_dataset_list, batch_size = batch_size, shuffle = True)
            test_loader = DataLoader(test_dataset_list, batch_size = 999, shuffle = False)
            val_loader = DataLoader(val_dataset_list, batch_size = batch_size, shuffle = True)
            df_record = pd.DataFrame(columns = ['epoch','train_RMSE_epoch','train_R2','val_RMSE_epoch','val_R2'])
            early_stopper = EarlyStopper()
            for epoch in range(1, 500):
                r_tl = 0.0
                r_vl = 0.0
                train_res_li = []
                train_pred_li = []
                train_expt_li = []
                val_res_li = []
                val_pred_li = []
                val_expt_li = []
                train_loss, train_r2, r_tl, train_res_li, train_pred_li, train_expt_li = train(r_tl, train_res_li, train_pred_li, train_expt_li)
                val_loss, val_r2, r_vl, val_res_li, val_pred_li, val_expt_li = val(val_loader, r_vl, val_res_li, val_pred_li, val_expt_li)
                df_record.loc[len(df_record.index)] = [epoch, r_tl, train_r2, r_vl, val_r2]
                print(f'Epoch: {epoch},train_loss:{round(float(r_tl),5)}, val_loss:{round(float(r_vl),5)}')
                if epoch > 5:
                    if early_stopper.early_stop(r_tl, r_vl, train_res_li, train_pred_li, train_expt_li, val_res_li, val_pred_li, val_expt_li):
                        break
            df_record.to_csv(f'{pth}/{model_name}/v8/dropp2p2_lr0005/layer_v3_pretrain/PHMD549/dpka_res_split_v3_10_freeze0/{model_name}_E{n_expt}f{fold}_training.csv', index = False)
