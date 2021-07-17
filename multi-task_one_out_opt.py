from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def build_model(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    inp = Dropout(0.5)(input_layer)
    _ = Dense(units=256)(inp)
    _ = BatchNormalization()(_)
    _ = Activation('relu')(_)
    bottleneck = Dropout(0.7)(_)

    outputs = []
    for i in range(output_dim):
        _ = Dense(units=32, activation='relu')(bottleneck)
        outputs.append(Dense(units=1, activation='sigmoid')(_))

    model = Model(inputs=input_layer, outputs=outputs)
    opt = SGD(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model
    
def get_loss(y_pred, y_true): # cross-entropy
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    N = y_pred.shape[0]
    return -np.sum((y_true*np.log(y_pred)) + ((1-y_true)*np.log(1-y_pred))) / N
    
def optimize(X_initial, model_up, model_dn, target_up, target_dn, scaler, h=0.00001, lr=0.01, early_stop=5):
    X_initial = scaler.transform(X_initial).reshape(1, -1)
    temp = X_initial
    min_loss_updated = copy.deepcopy(temp)
    min_ = 999
    cnt = 0
    gradient = np.zeros((1, 56))
    print('h:', h, 'lr:', lr)
    
    for k in range(0, 1000):
        for idx in range(56):
            x_1 = copy.deepcopy(temp)
            x_2 = copy.deepcopy(temp)
            x_1[:, idx] = x_1[:, idx] + h
        
            y_1_up = np.asarray([x1[0][0] for x1 in model_up.predict(x_1)])
            y_2_up = np.asarray([x2[0][0] for x2 in model_up.predict(x_2)])
        
            y_1_dn = np.asarray([x1[0][0] for x1 in model_dn.predict(x_1)])
            y_2_dn = np.asarray([x2[0][0] for x2 in model_dn.predict(x_2)])

            loss_1_up = get_loss(y_true=target_up, y_pred=y_1_up)
            loss_2_up = get_loss(y_true=target_up, y_pred=y_2_up)
        
            loss_1_dn = get_loss(y_true=target_dn, y_pred=y_1_dn)
            loss_2_dn = get_loss(y_true=target_dn, y_pred=y_2_dn)
        
            gradient[:, idx] = ((loss_1_up + loss_1_dn) - (loss_2_up + loss_2_dn)) / h
        
        temp = temp - lr*gradient
        
        y_up = np.asarray([x1[0][0] for x1 in model_up.predict(temp)])
        y_dn = np.asarray([x1[0][0] for x1 in model_dn.predict(temp)])
        
        loss_up = get_loss(y_true=target_up, y_pred=y_up)
        loss_dn = get_loss(y_true=target_dn, y_pred=y_dn)
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


descriptor = 'jtvae'
cell_line = 'A549'
obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
              random_genes=False, csv_file='Data/JTVAE_Representations.csv')

x_up, y_up, _ = obj.get_up_genes()
x_dn, y_dn, _ = obj.get_down_genes()
test_smiles = pd.read_csv('Data/test_smiles_a549.csv')['smiles'].values.tolist()
    
lst_y = []
for i in range(978):
    lst_y.append(np.count_nonzero(y_up.iloc[:, i]))
tmp_df = pd.DataFrame({'genes': lst_y})
indexes = tmp_df[tmp_df['genes'] >= 100].index.values
del lst_y, tmp_df
y_up = y_up[y_up.columns[indexes]]

lst_y = []
for i in range(978):
    lst_y.append(np.count_nonzero(y_dn.iloc[:, i]))
tmp_df = pd.DataFrame({'genes': lst_y})
indexes = tmp_df[tmp_df['genes'] >= 100].index.values
del lst_y, tmp_df
y_dn = y_dn[y_dn.columns[indexes]]
    
cikarilan_smi_lst = []
baslangic_smi_lst = []
distance_lst = []
optimized_smi_lst = []
    
for i in range(len(test_smiles)):
    idx_up = x_up[x_up['SMILES'] == test_smiles[i]].index[0]
    idx_dn = x_dn[x_dn['SMILES'] == test_smiles[i]].index[0]
    
    trn_x_up = x_up.drop(idx_up).drop(['SMILES'], axis=1).values.astype('float')
    trn_y_up = y_up.drop(idx_up).values.astype('int')
    trn_x_dn = x_dn.drop(idx_dn).drop(['SMILES'], axis=1).values.astype('float')
    trn_y_dn = y_dn.drop(idx_dn).values.astype('int')
    
    scaler = StandardScaler()
    scaler = scaler.fit(trn_x_up)
    trn_x_up = scaler.transform(trn_x_up)
    trn_x_dn = scaler.transform(trn_x_dn)
    
    test_y_up = y_up.loc[idx_up,:].values.astype('int')
    test_y_dn = y_dn.loc[idx_dn,:].values.astype('int')
    
    trn_y_up_list = []
    target_up = []
    for j in range(trn_y_up.shape[1]):
        trn_y_up_list.append(trn_y_up[:,j])
        target_up.append(test_y_up[j])
        
    trn_y_dn_list = []
    target_dn = []
    for j in range(trn_y_dn.shape[1]):
        trn_y_dn_list.append(trn_y_dn[:,j])
        target_dn.append(test_y_dn[j])

    print('Up Train shapes:', trn_x_up.shape, trn_y_up.shape)
    print('Dn Train shapes:', trn_x_dn.shape, trn_y_dn.shape)
    
    weights = []
    for trn_target in trn_y_up_list:
        w = class_weight.compute_class_weight('balanced', np.unique(trn_target), trn_target)
        weights.append({0: w[0], 1: w[1]})

    model_up = build_model(trn_x_up.shape[1], trn_y_up.shape[1])
    model_up.fit(trn_x_up, trn_y_up_list, batch_size=32, class_weight=weights, epochs=25, verbose=0)
    
    weights = []
    for trn_target in trn_y_dn_list:
        w = class_weight.compute_class_weight('balanced', np.unique(trn_target), trn_target)
        weights.append({0: w[0], 1: w[1]})

    model_dn = build_model(trn_x_dn.shape[1], trn_y_dn.shape[1])
    model_dn.fit(trn_x_dn, trn_y_dn_list, batch_size=32, class_weight=weights, epochs=25, verbose=0)
    
    pred_feat = x_up[x_up['SMILES'] == test_smiles[i]].drop(['SMILES'], axis=1).values.astype('float')
    
    arr = np.random.uniform(low=-1.5, high=1.5, size=(56,))
    baslangic_feat = pred_feat.flatten() + arr
    dec_smiles = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([baslangic_feat[0:28]])).float(),
                                          torch.from_numpy(np.asarray([baslangic_feat[28:56]])).float())
    
    cikarilan_smi = test_smiles[i]
    baslangic_smi = dec_smiles
    distance = np.linalg.norm(pred_feat - baslangic_feat)
    optimize_edilen_smi = ""
    
    print(cikarilan_smi, ':', baslangic_smi, ':', distance, ':', optimize_edilen_smi)
    
    pred_up = model_up.predict(scaler.transform(pred_feat))
    print('Iter:', str(i), 'pred_proba_up:', pred_up)
    
    pred_dn = model_dn.predict(scaler.transform(pred_feat))
    print('Iter:', str(i), 'pred_proba_dn:', pred_dn)
    
    pred_y_up = [0 if x[0][0] <= 0.5 else 1 for x in pred_up]
    pred_y_dn = [0 if x[0][0] <= 0.5 else 1 for x in pred_dn]
    
    if accuracy_score(target_up, pred_y_up) < 0.5 or accuracy_score(target_dn, pred_y_dn) < 0.5:
        K.clear_session()
        continue
    else:
        print(accuracy_score(target_up, pred_y_up), accuracy_score(target_dn, pred_y_dn))
        updated_feat = optimize(X_initial=baslangic_feat.reshape(1, -1),
                                model_up=model_up, model_dn=model_dn, 
                                target_up=target_up, target_dn=target_dn,
                                scaler=scaler).flatten()
        optimize_edilen_smi = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([updated_feat[0:28]])).float(),
                                                       torch.from_numpy(np.asarray([updated_feat[28:56]])).float())
                                                       
        cikarilan_smi_lst.append(cikarilan_smi)
        baslangic_smi_lst.append(baslangic_smi)
        distance_lst.append(distance)
        optimized_smi_lst.append(optimize_edilen_smi)
        
        print(cikarilan_smi, ':', baslangic_smi, ':', distance, ':', optimize_edilen_smi)
        K.clear_session()

df = pd.DataFrame({'cikarilan_smiles': cikarilan_smi_lst,
                   'baslangic_smiles': baslangic_smi_lst,
                   'distance': distance_lst,
                   'optimized_smiles': optimized_smi_lst})
df.to_csv('a549_multi_task_one_out_opt_15_iter_1000_lr_001.csv', index=False)
