import features
import sys 
from pycaret.datasets import get_data
import pickle
import pandas as pd
from pycaret.regression import setup, create_model, tune_model, evaluate_model, finalize_model, predict_model, compare_models, load_model
import numpy as np
import os

num_cols = ['com_min_d0_polar', 'com_min_d1_polar', 'com_min_d0_nonpolar', 'com_min_d1_nonpolar', 
            'com_min_d0_bb_NH', 'com_min_d1_bb_NH', 'com_min_d0_neg_O', 'com_min_d1_neg_O', 
            'com_min_d0_pos_N', 'com_min_d1_pos_N', 'com_min_d0_hbond_o', 'com_min_d1_hbond_o', 
            'com_min_d0_hbond_n', 'com_min_d1_hbond_n', 'com_min_d0_hbond_h', 'com_min_d1_hbond_h', 
            'com_min_d0_C_SG', 'com_min_d1_C_SG', 'com_n_hv_6', 'com_n_hv_9', 'com_n_hv_12', 'com_n_hv_15',
            'com_npolar_5', 'com_npolar_10', 'com_npolar_15', 
            'com_polar_5', 'com_polar_10', 'com_polar_15',
            'DA1', 'DA2', 'DD1', 'DD2', 'flexibility', 'n_sc_C', 'model_pka',"buried_ratio","metal"]

cat_cols = [ 'rss', 'rp2ss', 'rp4ss', 'rm2ss', 'rm4ss', "charge", 'group_code']

def read_pdb(file): 
    titr_res = ['ASP', 'GLU', 'CYS', 'HIS', 'TYR', 'LYS']
    pdbfile = open(file, 'r')
    #process pdb into AA chain 
    found_chain = False 
    orig_resnm = [] 
    orig_resid = [] 
    orig_chain = [] 
    for line in pdbfile: 
        #print(line) 
        x = line.split() 
        if x[0] == 'ATOM': 
            if len(x[4]) > 4: 
                next_resid = x[4][:-4]
                chain = x[4][0] 
            else: 
                next_resid = x[5]
                chain = x[4] 
        if x[0] == 'ATOM' and (len(orig_resid) == 0 or next_resid != orig_resid[-1]): 
            orig_resnm.append(x[3]) 
            orig_resid.append(next_resid) 
            orig_chain.append(chain) 
    i = 0

    while i < len(orig_resid) - 1: 
        if orig_resid[i] == orig_resid[i+1]:
            orig_resid.pop(i) 
            orig_resnm.pop(i) 
        else: 
            i += 1 
            

    is_res = [False] * len(orig_resnm) 
    for i in range(len(orig_resnm)): 
        if orig_resnm[i] in titr_res:
            is_res[i] = True 
    
    residues = [file] 
    for i in range(len(is_res)): 
        if is_res[i] == True: 
            s = [''] * 3 
            s[0] = orig_chain[i]
            s[1] = orig_resnm[i] 
            s[2] = orig_resid[i]
            residues.append(s) 
    return residues 





pdbfile = sys.argv[1] 
residues = read_pdb(pdbfile) 
print("Calculating features ...")
feats = features.generate(residues) 
with open('features.txt', 'w') as f: 
    f.write(feats)





df1=pd.read_csv(f"features.txt")
df1=df1[num_cols+cat_cols]

model_a=load_model("models/catboost_acid_finalized")
model_b=load_model("models/catboost_base_finalized")

print("Predicting ...")
pred_a=model_a.predict(df1)
pred_b=model_b.predict(df1)
os.system("rm error_record_100.txt")
os.system("rm ridainp")
os.system("rm X:A.dat")
os.system("rm rida_results.dat")
os.system("rm rida_results.tab")
os.system("rm rida_anchor2.tab")
os.system("rm logs.log")
os.system("rm features.txt")

out=open(pdbfile.split("/")[-1].split(".")[0]+"_pka.csv","w")
out.write("Residue, pKa"+"\n")
for k in range(len(residues)-1):
    if residues[k+1][1] in ["HIS","LYS"]:
        out.write(residues[k+1][1]+"-"+residues[k+1][2]+","+str(round(pred_b[k],3))+"\n")
    else:
        out.write(residues[k+1][1]+"-"+residues[k+1][2]+","+str(round(pred_a[k],3))+"\n")
out.close()


print("Done")



