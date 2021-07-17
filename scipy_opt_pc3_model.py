from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from get_data import GetData

import json
import pandas as pd
import numpy as np
import torch
import sys
import copy
import tensorflow as tf

from scipy.optimize import minimize

sys.path.append('/content/gdrive/My Drive/Optimizasyon-2')
from jtnn import *
sys.path.append('/content/gdrive/My Drive/Optimizasyon-2')

vocab = [x.strip("\r\n ") for x in open("/content/gdrive/My Drive/Optimizasyon-2/Data/unique_canonical_train_vocab.txt")]
vocab = Vocab(vocab)

hidden_size = 450
latent_size = 56
depth = 3
stereo = True

model_jtvae = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model_jtvae.load_state_dict(torch.load("/content/gdrive/My Drive/Optimizasyon-2/Model/model.iter-9-6000", map_location=torch.device('cpu')))  # opts.model_path

with open('./L1000CDS_subset.json', 'r') as f:
    L = json.load(f)
    
obj = GetData(L=L, cell_line='PC3', descriptor='jtvae', n_fold=5, random_state=42, random_genes=False, csv_file='Data/JTVAE_Representations.csv')
x, y, folds = obj.get_up_genes()
trn_x = x.drop(['SMILES'], axis=1).values.astype('float')

scaler = StandardScaler()
scaler.fit(trn_x)

cell_line = 'PC3'
gene_target_up = pd.read_csv('Data/harmonizome_dn_binarized_use_for_up_model.csv')
gene_target_dn = pd.read_csv('Data/harmonizome_up_binarized_use_for_dn_model.csv')

gene_target_up = gene_target_up[gene_target_up['disease1'] == 'Cancer of prostate_Prostate_GSE1413']
gene_target_dn = gene_target_dn[gene_target_dn['disease1'] == 'Cancer of prostate_Prostate_GSE1413']

dis_df = pd.read_csv('jtvae-hastalik-drug/prostate adenocarcinoma.csv') # PC3
model_up = load_model('Model/' + cell_line + '_multi_task_model_up.h5')
model_dn = load_model('Model/' + cell_line + '_multi_task_model_dn.h5')

file_name_up = 'Model/' + cell_line + '_multi_task_gene_list_up.txt'
f = open(file_name_up, 'r')
lines = f.readlines()
gene_list_up = [line.strip() for line in lines]
y_true_up = gene_target_up[gene_list_up].values[0]

file_name_dn = 'Model/' + cell_line + '_multi_task_gene_list_dn.txt'
f = open(file_name_dn, 'r')
lines = f.readlines()
gene_list_dn = [line.strip() for line in lines]
y_true_dn = gene_target_dn[gene_list_dn].values[0]

cikarilan_smi_lst = []
baslangic_smi_lst = []
distance_lst = []
optimized_smi_lst = []

def loss_func(par):
    y_up = np.asarray([x[0][0] for x in model_up.predict(np.asarray(par).reshape(1, -1))])
    y_dn = np.asarray([x[0][0] for x in model_dn.predict(np.asarray(par).reshape(1, -1))])

    loss_up = 1 - y_up[25]
    loss_dn = 1 - y_dn[25]
    
    return loss_up + loss_dn

    
def optimize(X_initial):
    X_initial = scaler.transform(X_initial.reshape(1, -1)).reshape(1, -1)

    res = minimize(loss_func, X_initial, method='SLSQP', options={'ftol': 1e-9, 'disp': True})
    
    return scaler.inverse_transform(res.x)
    
for idx in range(len(dis_df)):
    arr = np.random.uniform(low=-1.5, high=1.5, size=(56,))
    baslangic_feat = dis_df.values[idx, 3:].astype('float') + arr
    dec_smiles = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([baslangic_feat[0:28]])).float(),
                                          torch.from_numpy(np.asarray([baslangic_feat[28:56]])).float())
                                          
    cikarilan_smi = dis_df.values[idx, 2]
    baslangic_smi = dec_smiles                                
    distance = np.linalg.norm(dis_df.values[idx, 3:].astype('float') - baslangic_feat)
    optimize_edilen_smi = ""
    print(cikarilan_smi, ':', baslangic_smi, ':', distance, ':', optimize_edilen_smi)
                                          
    updated_feat = optimize(X_initial=baslangic_feat)
    optimize_edilen_smi = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([updated_feat[0:28]])).float(),
                                                   torch.from_numpy(np.asarray([updated_feat[28:56]])).float())
                                                   
    cikarilan_smi_lst.append(cikarilan_smi)
    baslangic_smi_lst.append(baslangic_smi)
    distance_lst.append(distance)
    optimized_smi_lst.append(optimize_edilen_smi)
    
    print(cikarilan_smi, ':', baslangic_smi, ':', distance, ':', optimize_edilen_smi)
        
df = pd.DataFrame({'cikarilan_smiles': cikarilan_smi_lst,
                   'baslangic_smiles': baslangic_smi_lst,
                   'distance': distance_lst,
                   'optimized_smiles': optimized_smi_lst})
                   
df.to_csv('prostate_pc3_multi_task_scipy_SLSQP_15.csv', index=False)
