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

sys.path.append('/content/gdrive/My Drive/Optimizasyon')
from jtnn import *
sys.path.append('/content/gdrive/My Drive/Optimizasyon')

vocab = [x.strip("\r\n ") for x in open("/content/gdrive/My Drive/Optimizasyon/Data/unique_canonical_train_vocab.txt")]
vocab = Vocab(vocab)

hidden_size = 450
latent_size = 56
depth = 3
stereo = True

model_jtvae = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model_jtvae.load_state_dict(torch.load("/content/gdrive/My Drive/Optimizasyon/Model/model.iter-9-6000", map_location=torch.device('cpu')))  # opts.model_path

with open('./L1000CDS_subset.json', 'r') as f:
    L = json.load(f)
    
obj = GetData(L=L, cell_line='MCF7', descriptor='jtvae', n_fold=5, random_state=42, random_genes=False, csv_file='Data/JTVAE_Representations.csv')
x, y, folds = obj.get_up_genes()
trn_x = x.drop(['SMILES'], axis=1).values.astype('float')

scaler = StandardScaler()
scaler.fit(trn_x)
    
def get_loss(predictions, targets): # cross-entropy
    if isinstance(targets, list):
        targets = np.asarray(targets)
    if isinstance(predictions, list):
        predictions = np.asarray(predictions)
    N = predictions.shape[0]
    return -np.sum((targets*np.log(predictions)) + ((1-targets)*np.log(1-predictions))) / N
    
def optimize(initial_list, model_up, model_dn, target_up, target_dn, h=0.00001, lr=0.02, early_stop=5):
    initial = scaler.transform(np.asarray(initial_list))
    temp = initial
    min_loss_updated = copy.deepcopy(temp)
    min_ = 999
    cnt = 0
    gradient = np.zeros((initial.shape[0], 56))
    print('h:', h, 'lr:', lr)
    
    target_up_list = []
    target_dn_list = []
    for _ in range(initial.shape[0]):
        target_up_list.append(target_up)
        target_dn_list.append(target_dn)
        
    print('Up target:', target_up_list[0])
    print('Dn target:', target_dn_list[0])

    for k in range(0, 1000):
        for idx in range(56):
            x_1 = copy.deepcopy(temp)
            x_2 = copy.deepcopy(temp)
            x_1[:, idx] = x_1[:, idx] + h
            
            pred_list = [x for x in model_up.predict(x_1)]
            y_1_up = np.zeros((pred_list[0].shape[0], len(pred_list)))
            for i in range(len(pred_list)):
                y_1_up[:, i] = pred_list[i].flatten()

            pred_list = [x for x in model_up.predict(x_2)]
            y_2_up = np.zeros((pred_list[0].shape[0], len(pred_list)))
            for i in range(len(pred_list)):
                y_2_up[:, i] = pred_list[i].flatten()
                
            pred_list = [x for x in model_dn.predict(x_1)]
            y_1_dn = np.zeros((pred_list[0].shape[0], len(pred_list)))
            for i in range(len(pred_list)):
                y_1_dn[:, i] = pred_list[i].flatten()
                
            pred_list = [x for x in model_dn.predict(x_2)]
            y_2_dn = np.zeros((pred_list[0].shape[0], len(pred_list)))
            for i in range(len(pred_list)):
                y_2_dn[:, i] = pred_list[i].flatten()
                
            for j in range(initial.shape[0]):
                loss_1_up = get_loss(predictions=y_1_up[j], targets=target_up_list[j])
                loss_2_up = get_loss(predictions=y_2_up[j], targets=target_up_list[j])
            
                loss_1_dn = get_loss(predictions=y_1_dn[j], targets=target_dn_list[j])
                loss_2_dn = get_loss(predictions=y_2_dn[j], targets=target_dn_list[j])
            
                gradient[j, idx] = ((loss_1_up + loss_1_dn) - (loss_2_up + loss_2_dn)) / h

        temp = temp - lr*gradient
        
        pred_list = [x for x in model_up.predict(temp)]
        y_up = np.zeros((pred_list[0].shape[0], len(pred_list)))
        for i in range(len(pred_list)):
            y_up[:, i] = pred_list[i].flatten()
            
        pred_list = [x for x in model_dn.predict(temp)]
        y_dn = np.zeros((pred_list[0].shape[0], len(pred_list)))
        for i in range(len(pred_list)):
            y_dn[:, i] = pred_list[i].flatten()
        
        loss_up = get_loss(predictions=y_up, targets=target_up_list)
        loss_dn = get_loss(predictions=y_dn, targets=target_dn_list)
        loss = loss_up + loss_dn
        
        print('Iter:', k, 'Loss:', loss)

        if min_ > loss:
            min_ = loss
            min_loss_updated = copy.deepcopy(temp)
            cnt = 0
        else:
            cnt += 1
            
        if cnt == early_stop:
            print("Early stopped.", 'Loss:' + str(loss))
            break
    
    return scaler.inverse_transform(min_loss_updated)


cell_line = 'MCF7'
gene_target_up = pd.read_csv('Data/harmonizome_dn_binarized_use_for_up_model.csv')
gene_target_dn = pd.read_csv('Data/harmonizome_up_binarized_use_for_dn_model.csv')

gene_target_up = gene_target_up[gene_target_up['disease2'] == 'Breast Cancer_3744']
gene_target_dn = gene_target_dn[gene_target_dn['disease2'] == 'Breast Cancer_3744']

dis_df = pd.read_csv('Data/approved_drug_for_breast_cancer_smiles_jtvae.csv') # MCF7
model_up = load_model('Model/' + cell_line + '_multi_task_model_up.h5')
model_dn = load_model('Model/' + cell_line + '_multi_task_model_dn.h5')

file_name_up = 'Model/' + cell_line + '_multi_task_gene_list_up.txt'
f = open(file_name_up, 'r')
lines = f.readlines()
gene_list_up = [line.strip() for line in lines]
gene_target_up = gene_target_up[gene_list_up].values

file_name_dn = 'Model/' + cell_line + '_multi_task_gene_list_dn.txt'
f = open(file_name_dn, 'r')
lines = f.readlines()
gene_list_dn = [line.strip() for line in lines]
gene_target_dn = gene_target_dn[gene_list_dn].values
    
cikarilan_smi_lst = []
baslangic_smi_lst = []
baslangic_features = []
distance_lst = []
optimized_smi_lst = []

print('Breast Cancer_3744')
    
for idx in range(len(dis_df)):
    if (dis_df.values[idx, 2] in cikarilan_smi_lst) or (len(dis_df.values[idx, 2]) > 105):
        continue
    arr = np.random.uniform(low=-1.5, high=1.5, size=(56,))
    baslangic_feat = dis_df.values[idx, 5:].astype('float') + arr
    dec_smiles = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([baslangic_feat[0:28]])).float(),
                                          torch.from_numpy(np.asarray([baslangic_feat[28:56]])).float())
    
    cikarilan_smi = dis_df.values[idx, 2]
    baslangic_smi = dec_smiles                                
    distance = np.linalg.norm(dis_df.values[idx, 5:].astype('float') - baslangic_feat)
    optimize_edilen_smi = ""
    print(cikarilan_smi, ':', baslangic_smi, ':', distance)
    
    cikarilan_smi_lst.append(cikarilan_smi)
    baslangic_smi_lst.append(baslangic_smi)
    baslangic_features.append(baslangic_feat)
    distance_lst.append(distance)
    
updated_feat = optimize(initial_list=baslangic_features,
                        model_up=model_up, model_dn=model_dn, 
                        target_up=gene_target_up[0], target_dn=gene_target_dn[0])
                   
for feature in updated_feat:
    optimize_edilen_smi = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([feature[0:28]])).float(),
                                                   torch.from_numpy(np.asarray([feature[28:56]])).float())
                                                   
    optimized_smi_lst.append(optimize_edilen_smi)
    print('optimized_smiles:', optimize_edilen_smi)

df = pd.DataFrame({'cikarilan_smiles': cikarilan_smi_lst,
                   'baslangic_smiles': baslangic_smi_lst,
                   'distance': distance_lst,
                   'optimized_smiles': optimized_smi_lst})
                   
df.to_csv('Breast_Cancer_3744_mcf7_multi_task_listed_input_15_lr_002_h_1e-5_max_1000.csv', index=False)
