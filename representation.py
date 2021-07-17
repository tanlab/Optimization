from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import model_from_yaml
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from optparse import OptionParser
import rdkit
import rdkit.Chem as Chem
import torch
import pandas as pd
import numpy as np
import sys
import subprocess
import os
sys.path.append('jtnn/')
from jtnn import *
sys.path.append('../')

class Representation:
   
    def __init__(self, descriptor='jtvae'):
        self.descriptor = descriptor
        self.loaded_model = None
        self.model = None
        
        if self.descriptor == 'ecfp_autoencoder':
            yaml_file = open('Models/autoencoder_optedilmemis.yaml', 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            self.loaded_model = model_from_yaml(loaded_model_yaml)
            # load weights into new model
            self.loaded_model.load_weights("Models/autoencoder_optedilmemis.h5")

        elif self.descriptor == 'jtvae':
            lg = rdkit.RDLogger.logger()
            lg.setLevel(rdkit.RDLogger.CRITICAL)

            # Jupyter notebookta hata verdiği için parser kapatıldı.
            # parser = OptionParser()
            # parser.add_option("-t", "--test", dest="test_path")
            # parser.add_option("-v", "--vocab", dest="vocab_path")
            # parser.add_option("-m", "--model", dest="model_path")
            # parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
            # parser.add_option("-l", "--latent", dest="latent_size", default=56)
            # parser.add_option("-d", "--depth", dest="depth", default=3)
            # parser.add_option("-e", "--stereo", dest="stereo", default=1)
            # opts, args = parser.parse_args()

            vocab = [x.strip("\r\n ") for x in open("unique_canonical_train_vocab.txt")]
            vocab = Vocab(vocab)

            hidden_size = 450
            latent_size = 56
            depth = 3
            stereo = True

            model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
            model.load_state_dict(torch.load("model.iter-9-6000", map_location=torch.device('cpu')))
            # opts.model_path #MPNVAE-h450-L56-d3-beta0.001/model.iter-4

            self.model = model

    def get_representation(self, smiles, descriptor, useChirality=False):
        representation = [smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return mol
        
        if descriptor == "jtvae":
            for i in self.jtvae_representation(smiles=smiles, mol=mol)[0]:
                representation.append(i)
            return np.asarray(representation)

        elif descriptor == "ecfp_autoencoder":
            for i in self.ecfp_representation(smiles=smiles, mol=mol):
                representation.append(i)
            return np.asarray(representation)

        elif descriptor == "topological":
            topological = Chem.RDKFingerprint(mol, fpSize=1024).ToBitString()
            topological = np.fromstring(topological,'u1') - ord('0')
            for i in topological:
                representation.append(i)
            return np.asarray(representation)

        elif descriptor == "ecfp":
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024,
                                                         useChirality=useChirality).ToBitString()
            ecfp = np.fromstring(ecfp,'u1') - ord('0')
            for i in ecfp:
                representation.append(i)
            return np.asarray(representation)

        elif descriptor == "maccs":
            maccs = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            maccs = np.fromstring(maccs,'u1') - ord('0')
            for i in maccs:
                representation.append(i)
            return np.asarray(representation)

        elif descriptor == "shed":
            smi = smiles
            mols = Chem.MolFromSmiles(smi) 
            hmols = Chem.AddHs(mols)
            AllChem.EmbedMolecule(hmols,AllChem.ETKDG())
            writer = Chem.SDWriter('smiles.sdf')
            hmols.SetProp("_SMILES","%s"%smi)
            writer.write(hmols)
            writer.close()
            shell = subprocess.call(["./shed.sh"])
            data = pd.read_csv("myout1.csv" ,delimiter = "\t",sep = ":", header= None)
            for i in data.columns.values:
                if str(data.iloc[0][i]).find(":")!=-1:
                    data.iloc[0][i]=data.iloc[0][i][str(data.iloc[0][i]).find(":")+1:]
                    representation.append(data.iloc[0][i])
            delete = subprocess.call(["./delete.sh"])
            return np.asarray(representation)

        elif descriptor == "cats2d":
            smi = smiles
            mols = Chem.MolFromSmiles(smi) 
            hmols = Chem.AddHs(mols)
            AllChem.EmbedMolecule(hmols,AllChem.ETKDG())
            writer = Chem.SDWriter('smiles.sdf')
            hmols.SetProp("_SMILES","%s"%smi)
            writer.write(hmols)
            writer.close()
            shell = subprocess.call(["./cats2d.sh"])
            data = pd.read_csv("myout1.csv" ,delimiter = "\t",sep = ":", header= None)
            for i in data.columns.values:
                if str(data.iloc[0][i]).find(":")!=-1:
                    data.iloc[0][i]=data.iloc[0][i][str(data.iloc[0][i]).find(":")+1:]
                    representation.append(data.iloc[0][i])
            subprocess.call(["./delete.sh"])
            return np.asarray(representation)

    
    def ecfp_representation(self, smiles, mol):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        a = np.asarray(fp1)
        # with a Sequential model
        get_3rd_layer_output = K.function([self.loaded_model.layers[0].input],
                                          [self.loaded_model.layers[1].output])
        layer_output = get_3rd_layer_output(np.asarray([a]))[0]
        
        return layer_output[0]

    def jtvae_representation(self, smiles, mol):
        koku = pd.DataFrame(columns=list(range(56)))
        
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=False)
        dec_smiles = self.model.reconstruct(smiles3D, DataFrame=koku)
        
        del smiles3D
        del dec_smiles
        torch.cuda.empty_cache()
        
        return koku.values
