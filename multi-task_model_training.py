from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

vocab = [x.strip("\r\n ") for x in open("Data/unique_canonical_train_vocab.txt")]
vocab = Vocab(vocab)

hidden_size = 450
latent_size = 56
depth = 3
stereo = True

model_jtvae = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model_jtvae.load_state_dict(torch.load("Model/model.iter-9-6000", map_location=torch.device('cpu')))  # opts.model_path

with open('./L1000CDS_subset.json', 'r') as f:
    L = json.load(f)
    
obj = GetData(L=L, cell_line='PC3', descriptor='jtvae', n_fold=5, random_state=42, random_genes=False, csv_file='Data/JTVAE_Representations.csv')

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


descriptor = 'jtvae'
for cell_line in ['A549', 'MCF7', 'PC3', 'VCAP']:
    for target in ['up', 'dn']:
        obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                        random_genes=False, csv_file='Data/JTVAE_Representations.csv')
        
        if target == 'up':
            x, y, folds = obj.get_up_genes()
        elif target == 'dn':
            x, y, folds = obj.get_down_genes()
            
        lst_y = []
        for i in range(978):
            lst_y.append(np.count_nonzero(y.iloc[:, i]))
        tmp_df = pd.DataFrame({'genes': lst_y})
        indexes = tmp_df[tmp_df['genes'] >= 100].index.values
        del lst_y, tmp_df
        y = y[y.columns[indexes]]
        
        trn_x = x.drop(['SMILES'], axis=1).values.astype('float')
        trn_y = y.values.astype('int')

        file_object = open(cell_line + '_multi_task_gene_list_' + target + '.txt', 'a')
        for y_col in y.columns:
            file_object.write(y_col + '\n')
        file_object.close()

        trn_y_list = []
        for j in range(trn_y.shape[1]):
            trn_y_list.append(trn_y[:,j])
        
        trn_x = StandardScaler().fit_transform(trn_x)
        print('Train shapes:', trn_x.shape, trn_y.shape)
        
        weights = []
        for trn_target in trn_y_list:
            w = class_weight.compute_class_weight('balanced', np.unique(trn_target), trn_target)
            weights.append({0: w[0], 1: w[1]})
    
        nn_model = build_model(trn_x.shape[1], trn_y.shape[1])
        nn_model.fit(trn_x, trn_y_list, batch_size=16, class_weight=weights, epochs=100, verbose=0)
        nn_model.save(cell_line + '_multi_task_model_' + target + '.h5')
        K.clear_session()
