import pandas as pd
import numpy as np
import sys
from biopandas.pdb import PandasPdb
import copy
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sys,os
import re

# process pdb ID and chain ID (unused) 
def process(ID,chain):
    a=pdbfixer.PDBFixer(pdbid=ID)
    a.findMissingResidues()
    print(a.findMissingResidues())
    a.findMissingAtoms()
    a.addMissingAtoms()
    b=list(a.topology.chains())
    remove=[]
    found=False
    for i in range(len(b)):
        if b[i].id==chain and not found:
            found=True
            continue
        remove.append(i)
    a.removeChains(remove)
    openmm.app.pdbfile.PDBFile.writeFile(topology=a.topology,positions=a.positions,file="./structure/"+ID+"_"+chain+"_fixed.pdb",keepIds=True)

# cal_min_dis_py functions 
def filter_one_atom_per_res(df, dis, resnb):
    df['dis'] = dis
    df = df[df['residue_number'] != resnb]
    df = df.sort_values(by = ['residue_number', 'dis'])
    df = df.drop_duplicates(subset = ['residue_number'], keep='first')
    return df

def filter_polar(df, resnb):
    df = df[df['residue_number'] != resnb]
    df = df.sort_values(by = ['residue_number', 'dis'])
    return df

def dis_polar(df, gc_coords):
    df['gc_x'] = gc_coords[0]
    df['gc_y'] = gc_coords[1]
    df['gc_z'] = gc_coords[2]
    df['dis'] = np.sqrt((df['x_coord'] - df['gc_x'])**2 + 
                 (df['y_coord'] - df['gc_y'])**2 + 
                 (df['z_coord'] - df['gc_z'])**2) 
    return df

def get_min1_min2(df):
    min1 = round(df['dis'].min(), 2)
    if len(df) > 1:
        min2 = round(df['dis'].nsmallest(2).iloc[-1], 2)
    else:
        min2 = 999
    return min1, min2

def cal_min_dis(pdbid, chain, resid, resnm, txt_id): 
    nan=' ,'
    resid = int(resid) 
    with open(f"error_record_{txt_id}.txt", "w") as err_txt:
        try:
            pth = f'/home/danko/Downloads/DeepKa-main/structure/{pdbid}_{chain}_fixed.pdb'
            gc_coords = []
            at1 = ''
            at2 = ''
            short_nm = ''
            if resnm == 'ASP':
                at1 = 'OD1'
                at2 = 'OD2'
                short_nm = 'D'
            elif resnm == 'GLU':
                at1 = 'OE1'
                at2 = 'OE2'
                short_nm = 'E'
            elif resnm == 'HIS':
                at1 = 'ND1'
                at2 = 'NE2'
                short_nm = 'H'    
            elif resnm == 'CYS':
                at1 = 'SG'
                at2 = 'SG'
                short_nm = 'C'
            elif resnm == 'LYS':
                at1 = 'NZ'
                at2 = 'NZ'
                short_nm = 'K'
            elif resnm == 'TYR':
                at1 = 'OH'
                at2 = 'OH'
                short_nm = 'Y'
            pdb_df = PandasPdb().read_pdb(pth)
            pdb_df.df['ATOM'] = pdb_df.df['ATOM'][pdb_df.df['ATOM']['element_symbol'] != 'H']
            at_df = pdb_df.df['ATOM'][(pdb_df.df['ATOM']['residue_name'] == resnm) & 
                                      (pdb_df.df['ATOM']['residue_number'] == resid) &
                                      (pdb_df.df['ATOM']['atom_name'].isin([at1, at2])) &
                                      (pdb_df.df['ATOM']['chain_id'] == chain)]
            if not at_df.empty:
                gc_coords = [at_df['x_coord'].mean(), at_df['y_coord'].mean(), at_df['z_coord'].mean()]
                DE_df = copy.deepcopy(pdb_df)
                RK_df = copy.deepcopy(pdb_df)
                TSY_df = copy.deepcopy(pdb_df)
                NQW_df = copy.deepcopy(pdb_df)
                H_df = copy.deepcopy(pdb_df)
                C_df = copy.deepcopy(pdb_df)
                nonpolar_df = copy.deepcopy(pdb_df)
                polar_df = copy.deepcopy(pdb_df)
                nn_NH_df = copy.deepcopy(pdb_df)
                hv_df = copy.deepcopy(pdb_df)
                DE_df.df['ATOM'] = DE_df.df['ATOM'] = DE_df.df['ATOM'][DE_df.df['ATOM']['residue_name'].isin(['ASP', 'GLU'])]
                DE_df.df['ATOM'] = DE_df.df['ATOM'][DE_df.df['ATOM']['atom_name'].isin(['OD1', 'OD2', 'OE1', 'OE2'])]
                RK_df.df['ATOM'] = RK_df.df['ATOM'] = RK_df.df['ATOM'][RK_df.df['ATOM']['residue_name'].isin(['ARG', 'LYS'])]
                RK_df.df['ATOM'] = RK_df.df['ATOM'][RK_df.df['ATOM']['atom_name'].isin(['NH1', 'NH2', 'NE', 'NZ'])]
                TSY_df.df['ATOM'] = TSY_df.df['ATOM'][TSY_df.df['ATOM']['residue_name'].isin(['THR', 'SER', 'TYR'])]
                TSY_df.df['ATOM'] = TSY_df.df['ATOM'][TSY_df.df['ATOM']['atom_name'].isin(['OG1', 'OG', 'OH'])]
                NQW_df.df['ATOM'] = NQW_df.df['ATOM'][NQW_df.df['ATOM']['residue_name'].isin(['ASN', 'GLN', 'TRP'])]
                NQW_df.df['ATOM'] = NQW_df.df['ATOM'][NQW_df.df['ATOM']['atom_name'].isin(['ND2', 'NE2', 'NE1'])]
                H_df.df['ATOM'] = H_df.df['ATOM'][H_df.df['ATOM']['residue_name'].isin(['HIS'])]
                H_df.df['ATOM'] = H_df.df['ATOM'][H_df.df['ATOM']['atom_name'].isin(['ND1', 'NE2'])]
                C_df.df['ATOM'] = C_df.df['ATOM'][C_df.df['ATOM']['residue_name'].isin(['CYS'])]
                C_df.df['ATOM'] = C_df.df['ATOM'][C_df.df['ATOM']['atom_name'].isin(['SG'])]
                nonpolar_df.df['ATOM'] = nonpolar_df.df['ATOM'][nonpolar_df.df['ATOM']['element_symbol'] == 'C']
                nn_NH_df.df['ATOM'] = nn_NH_df.df['ATOM'][nn_NH_df.df['ATOM']['atom_name'].isin(['N'])]
                polar_df = pdb_df.df['ATOM'][pdb_df.df['ATOM']['atom_name'].isin(['O', 'SD', 'N'])]
                df_tmp1 = pdb_df.df['ATOM'][(pdb_df.df['ATOM']['residue_name'] == 'ASN') &
                                            (pdb_df.df['ATOM']['atom_name'] == 'OD1')]
                df_tmp2 = pdb_df.df['ATOM'][(pdb_df.df['ATOM']['residue_name'] == 'GLN') &
                                            (pdb_df.df['ATOM']['atom_name'] == 'OE1')]
                polar_df = pd.concat([polar_df, df_tmp1, df_tmp2])
                if not DE_df.df['ATOM'].empty:
                    DE_dis = DE_df.distance(xyz = gc_coords, records = ('ATOM',))
                    DE_res_dis = filter_one_atom_per_res(DE_df.df['ATOM'], DE_dis, resid)
                    DE_min1, DE_min2 = get_min1_min2(DE_res_dis)
                else:
                    DE_min1 = DE_min2 = 999
                if not RK_df.df['ATOM'].empty:
                    RK_dis = RK_df.distance(xyz = gc_coords, records = ('ATOM',))
                    RK_res_dis = filter_one_atom_per_res(RK_df.df['ATOM'], RK_dis, resid)
                    RK_min1, RK_min2 = get_min1_min2(RK_res_dis)
                else:
                    RK_min1 = RK_min2 = 999
                if not TSY_df.df['ATOM'].empty:
                    TSY_dis = TSY_df.distance(xyz = gc_coords, records = ('ATOM',))
                    TSY_res_dis = filter_one_atom_per_res(TSY_df.df['ATOM'], TSY_dis, resid)
                    TSY_min1, TSY_min2 = get_min1_min2(TSY_res_dis)
                else:
                    TSY_min1 = TSY_min2 = 999
                if not NQW_df.df['ATOM'].empty:
                    NQW_dis = NQW_df.distance(xyz = gc_coords, records = ('ATOM',))
                    NQW_res_dis = filter_one_atom_per_res(NQW_df.df['ATOM'], NQW_dis, resid)
                    NQW_min1, NQW_min2 = get_min1_min2(NQW_res_dis)
                else:
                    NQW_min1 = NQW_min2 = 999
                if not H_df.df['ATOM'].empty:
                    H_dis = H_df.distance(xyz = gc_coords, records = ('ATOM',))
                    H_res_dis = filter_one_atom_per_res(H_df.df['ATOM'], H_dis, resid)
                    H_min1, H_min2 = get_min1_min2(H_res_dis)
                else:
                    H_min1 = H_min2 = 999
                if not C_df.df['ATOM'].empty:
                    C_dis = C_df.distance(xyz = gc_coords, records = ('ATOM',))
                    C_res_dis = filter_one_atom_per_res(C_df.df['ATOM'], C_dis, resid)
                    C_min1, C_min2 = get_min1_min2(C_res_dis)
                else:
                    C_min1 = C_min2 = 999
                if not polar_df.empty:
                    polar_df = dis_polar(polar_df, gc_coords)
                    polar_res_dis = filter_polar(polar_df, resid)
                    polar_min1, polar_min2 = get_min1_min2(polar_res_dis)
                    n_polar_5 = len(polar_res_dis[polar_res_dis['dis'] <= 5])
                    n_polar_10 = len(polar_res_dis[polar_res_dis['dis'] <= 10])
                    n_polar_15 = len(polar_res_dis[polar_res_dis['dis'] <= 15])
                    #print(polar_res_dis[polar_res_dis['dis'] <= 15])
                    #print(polar_res_dis)
                if not nonpolar_df.df['ATOM'].empty:
                    nonpolar_dis = nonpolar_df.distance(xyz = gc_coords, records = ('ATOM',))
                    #print(nonpolar_df)
                    nonpolar_df = nonpolar_df.df['ATOM']
                    nonpolar_df['dis'] = nonpolar_dis
                    nonpolar_res_dis = filter_polar(nonpolar_df, resid)
                    nonpolar_min1, nonpolar_min2 = get_min1_min2(nonpolar_res_dis)
                    n_nonpolar_5 = len(nonpolar_res_dis[nonpolar_res_dis['dis'] <= 5])
                    n_nonpolar_10 = len(nonpolar_res_dis[nonpolar_res_dis['dis'] <= 10])
                    n_nonpolar_15 = len(nonpolar_res_dis[nonpolar_res_dis['dis'] <= 15])
                if not nn_NH_df.df['ATOM'].empty:
                    nn_NH_dis = nn_NH_df.distance(xyz = gc_coords, records = ('ATOM',))
                    nn_NH_res_dis = filter_one_atom_per_res(nn_NH_df.df['ATOM'], nn_NH_dis, resid)
                    nn_NH_min1, nn_NH_min2 = get_min1_min2(nn_NH_res_dis)
                if not hv_df.df['ATOM'].empty:
                    hv_dis = hv_df.distance(xyz = gc_coords, records = ('ATOM',))
                    hv_res_df = hv_df.df['ATOM']
                    hv_res_df['dis'] = hv_dis
                    hv_res_df = hv_res_df[hv_res_df['residue_number'] != resid]
                    n_hv_6 = len(hv_res_df[hv_res_df['dis'] <= 6])
                    n_hv_9 = len(hv_res_df[hv_res_df['dis'] <= 9])
                    n_hv_12 = len(hv_res_df[hv_res_df['dis'] <= 12])
                    n_hv_15 = len(hv_res_df[hv_res_df['dis'] <= 15])
                return f'{polar_min1}, {polar_min2}, {nonpolar_min1}, {nonpolar_min2}, {nn_NH_min1}, {nn_NH_min2}, {DE_min1}, {DE_min2}, {RK_min1}, {RK_min2}, {TSY_min1}, {TSY_min2}, {NQW_min1}, {NQW_min2}, {H_min1}, {H_min2}, {C_min1}, {C_min2}, {n_hv_6}, {n_hv_9}, {n_hv_12}, {n_hv_15}, {n_nonpolar_5},{n_nonpolar_10},{n_nonpolar_15},{n_polar_5},{n_polar_10},{n_polar_15}'
            else:
                err_txt.write(f'{pdbid}-{chain}-{resid}-{resnm}, atom not in PDB file')
                return f'{nan}' * 28 
        except FileNotFoundError:
            err_txt.write(f'{pdbid}-{chain}-{resid}-{resnm}, PDB file not exist')
            return f'{nan}' * 28
            pass
        except ValueError:
            err_txt.write(f'{pdbid}-{chain}-{resid}-{resnm}, Value error')
            return f'{nan}' * 28 
            pass
    err_txt.close()


#pdb2fasta and flex scripts as functions 
def pdb2fasta(pdb_files,pdbid,chains):

    fastaout=open(f"seq_{pdbid}_{chains}.fasta","w")
    aa3to1={
   'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
   'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
   'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
   'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
   'MSE':'M',
    }

    ca_pattern=re.compile("^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])|^HETATM\s{0,4}\d{1,5}\s{2}CA\s[\sA](MSE)\s([\s\w])")
    for pdb_file in [pdb_files]:
        filename=os.path.basename(pdb_file).split('.')[0]
        chain_dict=dict()
        chain_list=[]

        fp=open(pdb_file,'r')
        for line in fp.read().splitlines():
            if line.startswith("ENDMDL"):
                break
            match_list=ca_pattern.findall(line)
            if match_list:
                resn=match_list[0][0]+match_list[0][2]
                chain=match_list[0][1]+match_list[0][3]
                if chain in chain_dict:
                    chain_dict[chain]+=aa3to1[resn]
                else:
                    chain_dict[chain]=aa3to1[resn]
                    chain_list.append(chain)
        fp.close()

        for chain in chain_list:
            sys.stdout.write('>%s:%s\n%s\n'%(filename,chain,chain_dict[chain]))
        fastaout.write('>%s:%s\n%s\n'%(filename,chain,chain_dict[chain]))
        fastaout.close()


def flex(uid,resid):
    with open(uid, 'r') as filen:
        next(filen)
        text = ''
        for line in filen:
            text = text + line
        text = text.replace('\n','')
    
        if True:#((int(resid) >= 5) and (int(resid) <= len(text)-4)):
        
            tmp_seq =  text[int(resid)-16:int(resid)+16]
            if int(resid)<15:
            
                tmp_seq =  text[:32]
            if int(resid)>len(text)-16:
                tmp_seq = text[-32:]
            H_seq = tmp_seq + '0'
            out=open("ridainp","w")
            out.write(">X:A"+"\n")
            out.write(tmp_seq+"\n")
            out.close()
            os.system(f"/home/danko/Downloads/RIDA/rida ridainp")
            in1=np.loadtxt("X:A.dat",comments=["#"],usecols = (9))
            print(in1,np.mean(in1))
            analysed_seq = ProteinAnalysis(H_seq)
            f = analysed_seq.flexibility()
        
            flexibility = np.mean(in1)# f[0]
        else:
            flexibility = ''
        return flexibility

def coor_dat(chain, a1rid, a2rid, d1rid, d2rid, resid, filen): 
    print(chain,a1rid,a2rid,d1rid,d2rid,resid,filen)

    out1=open(chain+"_"+resid+"_bbha1_coor.dat","w")
    out2=open(chain+"_"+resid+"_bbha2_coor.dat","w")

    out3=open(chain+"_"+resid+"_bbhd1_coor.dat","w")
    out4=open(chain+"_"+resid+"_bbhd2_coor.dat","w")

    out5=open(chain+"_"+resid+"_bbhd1H_coor.dat","w")
    out6=open(chain+"_"+resid+"_bbhd2H_coor.dat","w")

    out7=open(chain+"_"+resid+"_O_coor.dat","w")
    out8=open(chain+"_"+resid+"_H_coor.dat","w")
    out9=open(chain+"_"+resid+"_N_coor.dat","w")

    in1=tuple(open(filen,"r"))
    for i in range(len(in1)):
        if in1[i][0:4]=="ATOM":        

            l=in1[i].split()

            if l[2]=="O" and l[5]==str(a1rid):
                out1.write(in1[i])
            if l[2]=="O" and l[5]==str(a2rid):
                out2.write(in1[i])

            if l[2]=="N" and l[5]==str(d1rid):
                out3.write(in1[i])
            if l[2]=="N" and l[5]==str(d2rid):
                out4.write(in1[i])

            if l[2]=="H" and l[5]==str(d1rid):
                out5.write(in1[i])
            if l[2]=="H" and l[5]==str(d2rid):
                out6.write(in1[i])

            if l[2]=="O" and l[5]==str(resid):
                out7.write(in1[i])

            if l[2]=="H" and l[5]==str(resid):
                out8.write(in1[i])
            if l[2]=="N" and l[5]==str(resid):
                out9.write(in1[i])

    out1.close()             
    out2.close()
    out3.close()
    out4.close()
    out5.close()
    out6.close()
    out7.close()
    out8.close()
    out9.close()

def bbh_dis(atom, chain, resid, role):
    df = pd.read_csv(f'{atom}_{chain}_{resid}_{role}_coor.dat', delim_whitespace = True, header=None)
    pd.to_numeric(df[2])
    pd.to_numeric(df[3])
    pd.to_numeric(df[1])
    pd.to_numeric(df[4])
    pd.to_numeric(df[5])
    pd.to_numeric(df[0])
    df['xdiff']= df[0] - df[3]
    df['ydiff']= df[1] - df[4]
    df['zdiff']= df[2] - df[5]
    df['dis'] = np.sqrt(df['xdiff']**2 + df['ydiff']**2 + df['zdiff']**2)
    df.to_csv(f'{atom}_{chain}_{resid}_{role}_dis.dat', index=False)

def to_float(x): 
    if x.strip() == '':
        return ''
    else: 
        return float(x) 

infile = sys.argv[1] 
outfile = sys.argv[2] 
out = open(outfile, "w")
pdbdir='/home/danko/Downloads/DeepKa-main/structure/'

print("PDB_ID,Chain,Res_Name,Res_ID,Expt_pKa,Uniprot_ID,info,com_min_d0_polar,com_min_d1_polar,com_min_d0_nonpolar,com_min_d1_nonpolar,com_min_d0_bb_NH,com_min_d1_bb_NH,com_min_d0_neg_O,com_min_d1_neg_O,com_min_d0_pos_N,com_min_d1_pos_N,com_min_d0_hbond_o,com_min_d1_hbond_o,com_min_d0_hbond_n,com_min_d1_hbond_n,com_min_d0_hbond_h,com_min_d1_hbond_h,com_min_d0_C_SG,com_min_d1_C_SG,com_n_hv_6,com_n_hv_9,com_n_hv_12,com_n_hv_15,com_npolar_5,com_npolar_10,com_npolar_15,com_polar_5,com_polar_10,com_polar_15,ats,rss,rp2ss,rp4ss,rm2ss,rm4ss,DA1,DA2,DD1,DD2,d_fe,d_zn,d_mg,flexibility", file = out) 

# read input.txt file 
f = open(infile, 'r') 
x = f.read().split() 
x = x[0:]

#process all pdb files
'''
for line in x: 
    s = line.split(',') 
    pdbid = s[0] 
    chain = s[1] 
    process(pdbid, chain) 
'''

# generate features
for L in x: 
    s = L.split(',') 
    pdbid = s[0] 
    chain = s[1] 
    resnm = s[2] 
    resid = s[3] 
    pKa = s[4] 
    uniid = s[5] 
    
    #######################################
    ## test if the fixed PDB file exists ##
    #######################################
    filen = 'Not exist' 
    spalen = len(resid) 
    spa = ' ' * (4-spalen) 
    
    pdbpattern = re.compile(f'ATOM.*CA.*{resnm} {chain}{spa}{resid}')
    for line in open(f'{pdbdir}{pdbid}_{chain}_fixed.pdb', 'r'): 
        if re.search(pdbpattern, line) != None: 
            filen = f'{pdbdir}{pdbid}_{chain}_fixed.pdb' 
            break 
    print(filen) 
    
    #atom1 and atom2 are obsolete? 
    if resnm== "HIS": 
        atom1='ND1'
        atom2='NE2'
        resn='H'
    elif resnm== "ASP": 
        atom1='OD1'
        atom2='O0D2'
        resn='D'
    elif resnm == "GLU":
        atom1='OE1'
        atom2='OE2'
        resn='E'
    elif resnm == "CYS":
        atom1='SG'
        atom2='SG'
        resn='C'
    elif resnm == "LYS":
        atom1='NZ'
        atom2='NZ'
        resn='K'
    elif resnm == "TYR":
        atom1='OH'
        atom2='OH'
        resn='Y'
        
    ##########################################
    ## calculate distance to various groups ##
    ##########################################
    
    pyout = cal_min_dis(pdbid, chain, resid, resnm, '100') 
    #print(pyout) 
    #run DSSP and read results 
    os.system(f"/home/danko/Downloads/mkdssp  {pdbdir}{pdbid}_{chain}_fixed.pdb  {pdbid}_{chain}_fixed.dssp")
    dssp_file=tuple(open(f"{pdbid}_{chain}_fixed.dssp"))

    # run through dssp file for residue row
    print(f'{resid} {chain} {resn}')
    res_row = 0
    for i in range(len(dssp_file)):
        if dssp_file[i].find(f'{resid} {chain} {resn}') != -1: 
            res_row = i 
            break 
    res_dssp = dssp_file[res_row]
    asa = float(res_dssp[35:38]) #accessible surface area 
    #print(asa) 
    #protein secondary structure of CYS and residues that are 2 and 4 AAs away from CYS 
    rss = res_dssp[16] 
    if res_row+2 < len(dssp_file):
        rp2ss = dssp_file[res_row+2][16]
    else: 
        rp2ss = ' '
    if res_row+4 < len(dssp_file):
        rp4ss = dssp_file[res_row+4][16]
    else: 
        rp4ss = ' '
    rm2ss = dssp_file[res_row-2][16]
    rm4ss = dssp_file[res_row-4][16] 
    
    bhba1 = int(res_dssp[40:45])
    bhbd1 = int(res_dssp[51:56])
    bhba2 = int(res_dssp[62:67])
    bhbd2 = int(res_dssp[73:78])
    
    a1rid = bhba1 + int(resid) 
    d1rid = bhbd1 + int(resid) 
    a2rid = bhba2 + int(resid) 
    d2rid = bhbd2 + int(resid) 
    #run coor_dat 
    coor_dat(chain, a1rid, a2rid, d1rid, d2rid, resid, filen) 
    ocor = open(f'{chain}_{resid}_O_coor.dat', 'r').read()
    ox = to_float(ocor[30:38])
    oy = to_float(ocor[38:46])
    oz = to_float(ocor[46:54]) 
    ncor = open(f'{chain}_{resid}_N_coor.dat', 'r').read()
    nx = to_float(ncor[30:38]) 
    ny = to_float(ncor[38:46]) 
    nz = to_float(ncor[46:54]) 
    hcor = open(f'{chain}_{resid}_H_coor.dat', 'r').read()
    hx = to_float(hcor[30:38]) 
    hy = to_float(hcor[38:46]) 
    hz = to_float(hcor[46:54]) 
    
    # distance features from coor_dat and bbh_dis (DA1, DA2, DD1, DD2) 
    in2 = open(f'{chain}_{resid}_bbha1_coor.dat', 'r')
    a1cor = in2.read()
    a1x = to_float(a1cor[30:38]) 
    a1y = to_float(a1cor[38:46]) 
    a1z = to_float(a1cor[46:54]) 
    out2 = open(f'{resn}_{chain}_{resid}_bbha1_coor.dat', 'w') 
    out2.write(f'{nx} {ny} {nz} {a1x} {a1y} {a1z}')
    print(f'{nx} {ny} {nz} {a1x} {a1y} {a1z}')
    out2.close()
    if a1x != '' and a1y != '' and a1z != '': 
        bbh_dis(resn, chain, resid, 'bbha1') 
        la1 = open(f'{resn}_{chain}_{resid}_bbha1_dis.dat', 'r').readlines()
        aa1 = [row.split(',') for row in la1]
        DA1 = aa1[1][9].strip()
    
    
    in3 = open(f'{chain}_{resid}_bbha2_coor.dat', 'r')
    a2cor = in3.read() 
    a2x = to_float(a2cor[30:38])
    a2y = to_float(a2cor[38:46]) 
    a2z = to_float(a2cor[46:54])
    out3 = open(f'{resn}_{chain}_{resid}_bbha2_coor.dat', 'w')
    out3.write(f'{nx} {ny} {nz} {a2x} {a2y} {a2z}')
    print(f'{nx} {ny} {nz} {a2x} {a2y} {a2z}')
    out3.close()
    if a2x != '' and a2y != '' and a2z != '': 
        bbh_dis(resn, chain, resid, 'bbha2') 
        la2 = open(f'{resn}_{chain}_{resid}_bbha2_dis.dat', 'r').readlines()
        aa2 = [row.split(',') for row in la2]
        DA2 = aa2[1][9].strip()
    
    
    in4 = open(f'{chain}_{resid}_bbhd1_coor.dat', 'r')
    d1cor = in4.read()
    d1x = to_float(d1cor[30:38])
    d1y = to_float(d1cor[38:46]) 
    d1z = to_float(d1cor[46:54]) 
    out4 = open(f'{resn}_{chain}_{resid}_bbhd1_coor.dat', 'w') 
    out4.write(f'{ox} {oy} {oz} {d1x} {d1y} {d1z}')
    out4.close()
    if d1x != '' and d1y != '' and d1z != '':
        bbh_dis(resn, chain, resid, 'bbhd1') 
        ld1 = open(f'{resn}_{chain}_{resid}_bbhd1_dis.dat', 'r').readlines()
        ad1 = [row.split(',') for row in ld1]
        DD1 = ad1[1][9].strip()
    
    
    in5 = open(f'{chain}_{resid}_bbhd2_coor.dat', 'r')
    d2cor = in5.read()
    d2x = to_float(d2cor[30:38]) 
    d2y = to_float(d2cor[38:46]) 
    d2z = to_float(d2cor[46:54]) 
    out5 = open(f'{resn}_{chain}_{resid}_bbhd2_coor.dat', 'w')
    out5.write(f'{ox} {oy} {oz} {d2x} {d2y} {d2z}')
    out5.close()
    if d2x != '' and d2y != '' and d2z != '':
        bbh_dis(resn, chain, resid, 'bbhd2') 
        ld2 = open(f'{resn}_{chain}_{resid}_bbhd2_dis.dat', 'r').readlines()
        ad2 = [row.split(',') for row in ld2]
        DD2 = ad2[1][9].strip()
    
    print(DA1, DA2, DD1, DD2)
    
    com_min_d0_fe=999
    com_min_d0_zn=999
    com_min_d0_mg=999
    
    #biopython flexibility 
    pdb2fasta(f'{pdbdir}{pdbid}_{chain}_fixed.pdb', pdbid, chain) 
    flexi = flex(f"seq_{pdbid}_{chain}.fasta",resid) 
    
    #final outfile print 
    info = f'{pdbid}-{chain}-{resid}'
    out.write(f'{L}, {info}, {pyout}, {asa}, {rss}, {rp2ss}, {rp4ss}, {rm2ss}, {rm4ss}, {DA1}, {DA2}, {DD1}, {DD2}, {com_min_d0_fe}, {com_min_d0_zn}, {com_min_d0_mg}, {flexi}\n')
    
    # remove created files 
    maindir = '/home/danko/Downloads/DeepKa-main'
    test = os.listdir(maindir) 
    for item in test: 
        if item.endswith('.dat') or item.endswith('.dssp') or item.endswith('.asa') or item.endswith('.rsa') or item.endswith('.log') or item.endswith('.fasta'): 
            os.remove(os.path.join(maindir, item))
    
out.close() 
