from sklearn.model_selection import KFold
from rdkit import Chem
from representation import Representation
import pandas as pd
import numpy as np
import sys


class GetData:
    
    def __init__(self, L, cell_line, descriptor='jtvae', n_fold=5, random_state=0, random_genes=False,
                 csv_file="", useChirality=False):
        """
            Parameters
            -----------
            L: dictionary from L1000CDS_subset.json

            cell_line: cell_id
                params:
                    string: 'VCAP', 'A549', 'A375', 'PC3', 'MCF7', 'HT29', etc.

            descriptor: descriptor for chemical compounds.
                params:
                    string: 'ecfp', 'ecfp_autoencoder', 'maccs', 'topological', 'shed', 'cats2d', 'jtvae'(default)

            n_fold: number of folds
                params:
                    int: 5(default)

            random_state: random_state for Kfold
                params:
                    int: 0(default)

            random_genes: if it is true, returns random 20 genes from target values
                params:
                    bool: False(default)
                    list of random genes: [118, 919, 274, 866, 354, 253, 207, 667, 773, 563,
                                           553, 918, 934, 81, 56, 232, 892, 485, 30, 53]

            csv_file: if it is not empty, representation data used from this file
                params:
                    string: "<csv_file_path>"
        """
        
        self.L = L
        self.cell_line = cell_line
        self.descriptor = descriptor
        self.n_fold = n_fold
        self.random_state = random_state
        self.random_genes = random_genes
        self.csv_file = csv_file
        self.useChirality = useChirality
        
        if self.useChirality and self.descriptor != 'ecfp':
            sys.exit('useChirality parameter is only usable with ecfp descriptor.')
            
        self.random_index_list = [118, 919, 274, 866, 354, 253, 207, 667, 773, 563,
                                  553, 918, 934, 81, 56, 232, 892, 485, 30, 53]

        self.LmGenes = []
        self.meta_smiles = pd.read_csv('meta_SMILES.csv')

        file_path = 'LmGenes.txt'
        with open(file_path) as fp:
            line = fp.readline()
            while line:
                self.LmGenes.append(line.strip())
                line = fp.readline()
        
        self.rep = Representation(self.descriptor)

    def get_regression_data(self):
        X = []
        Y = []
        perts = []
        unique_smiles = []
        counter = 0
        length = len(self.L[self.cell_line])
        
        print('Getting data...')
        
        data = None
        if len(self.csv_file) != 0:
            data = pd.read_csv(self.csv_file)

        for pert_id in self.L[self.cell_line]:
            counter += 1
            if counter % 10 == 0:
                print('%.1f %%    \r' % (counter / length * 100), end=""),
                
            smiles = self.meta_smiles[self.meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
            if str(smiles) == 'nan' or str(smiles) == '-666':
                continue

            if not self.useChirality:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                canonical_smiles = smiles

            if canonical_smiles in unique_smiles or len(canonical_smiles) > 120:
                continue
            if data is not None:
                if data[data['pert_id'] == pert_id].empty:
                    continue
                else:
                    feature = data[data['pert_id'] == pert_id].drop(['pert_id'], axis=1).values[0].tolist()
            else:
                feature = self.rep.get_representation(smiles=canonical_smiles, descriptor=self.descriptor,
                                                      useChirality=self.useChirality)

            unique_smiles.append(canonical_smiles)
            labels = self.L[self.cell_line][pert_id]['chdirLm']
            X.append(feature)
            Y.append(labels)
            perts.append(pert_id)

        x = np.asarray(X)
        y = np.asarray(Y)

        x_columns = ['SMILES']
        if self.descriptor == 'ecfp':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_' + str(i + 1))
        elif self.descriptor == 'ecfp_autoencoder':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_autoencoder_' + str(i + 1))
        elif self.descriptor == 'topological':
            for i in range(x.shape[1]-1):
                x_columns.append('topological_' + str(i + 1))
        elif self.descriptor == 'maccs':
            for i in range(x.shape[1]-1):
                x_columns.append('maccs_' + str(i + 1))
        elif self.descriptor == 'jtvae':
            for i in range(x.shape[1]-1):
                x_columns.append('jtvae_' + str(i + 1))
        elif self.descriptor == 'shed':
            for i in range(x.shape[1]-1):
                x_columns.append('shed_' + str(i + 1))
        elif self.descriptor == 'cats2d':
            for i in range(x.shape[1]-1):
                x_columns.append('cats2d_' + str(i + 1))

        x = pd.DataFrame(x, index=perts, columns=x_columns)
        y = pd.DataFrame(y, index=perts)
        folds = list(KFold(self.n_fold, shuffle=True, random_state=self.random_state).split(x))

        if self.random_genes:
            y_random = []
            for i in self.random_index_list:
                y_random.append(y.iloc[:, i:i + 1])
            df = y_random[0]
            for i in range(len(y_random) - 1):
                df = pd.concat([df, y_random[i + 1]], axis=1)
            y = df

        print('\nDone.')
        
        return x, y, folds

    def get_up_genes(self):
        X = []
        Y = []
        perts = []
        unique_smiles = []
        counter = 0
        length = len(self.L[self.cell_line])
        
        print('Getting data...')
        
        class_dict = {}
        data = None
        if len(self.csv_file) != 0:
            data = pd.read_csv(self.csv_file)

        for gene in self.LmGenes:
            class_dict.update({gene: 0})

        for pert_id in self.L[self.cell_line]:
            counter += 1
            if counter % 10 == 0:
                print('%.1f %%    \r' % (counter / length * 100), end=""),
                
            if 'upGenes' not in self.L[self.cell_line][pert_id]:
                continue

            smiles = self.meta_smiles[self.meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
            if str(smiles) == 'nan' or str(smiles) == '-666':
                continue

            if not self.useChirality:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                canonical_smiles = smiles

            if canonical_smiles in unique_smiles or len(canonical_smiles) > 120:
                continue
            if data is not None:
                if data[data['pert_id'] == pert_id].empty:
                    continue
                else:
                    feature = data[data['pert_id'] == pert_id].drop(['pert_id'], axis=1).values[0].tolist()
            else:
                feature = self.rep.get_representation(smiles=canonical_smiles, descriptor=self.descriptor,
                                                      useChirality=self.useChirality)

            unique_smiles.append(canonical_smiles)
            up_genes = list(set(self.L[self.cell_line][pert_id]['upGenes']))
            class_dict = dict.fromkeys(class_dict, 0)

            for gene in up_genes:
                if gene in class_dict:
                    class_dict.update({gene: 1})

            labels = np.fromiter(class_dict.values(), dtype=int)
            X.append(feature)
            Y.append(labels)
            perts.append(pert_id)

        x = np.asarray(X)
        y = np.asarray(Y)

        x_columns = ['SMILES']
        y_columns = list(class_dict.keys())
        if self.descriptor == 'ecfp':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_' + str(i + 1))
        elif self.descriptor == 'ecfp_autoencoder':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_autoencoder_' + str(i + 1))
        elif self.descriptor == 'topological':
            for i in range(x.shape[1]-1):
                x_columns.append('topological_' + str(i + 1))
        elif self.descriptor == 'maccs':
            for i in range(x.shape[1]-1):
                x_columns.append('maccs_' + str(i + 1))
        elif self.descriptor == 'jtvae':
            for i in range(x.shape[1]-1):
                x_columns.append('jtvae_' + str(i + 1))
        elif self.descriptor == 'shed':
            for i in range(x.shape[1]-1):
                x_columns.append('shed_' + str(i + 1))
        elif self.descriptor == 'cats2d':
            for i in range(x.shape[1]-1):
                x_columns.append('cats2d_' + str(i + 1))

        x = pd.DataFrame(x, index=perts, columns=x_columns)
        y = pd.DataFrame(y, index=perts, columns=y_columns)
        folds = list(KFold(self.n_fold, shuffle=True, random_state=self.random_state).split(x))

        if self.random_genes:
            y_random = []
            for i in self.random_index_list:
                y_random.append(y.iloc[:, i:i + 1])
            df = y_random[0]
            for i in range(len(y_random) - 1):
                df = pd.concat([df, y_random[i + 1]], axis=1)
            y = df

        print('\nDone.')
        
        return x, y, folds

    def get_down_genes(self):
        X = []
        Y = []
        perts = []
        unique_smiles = []
        counter = 0
        length = len(self.L[self.cell_line])
        
        print('Getting data...')
        
        class_dict = {}
        data = None
        if len(self.csv_file) != 0:
            data = pd.read_csv(self.csv_file)

        for gene in self.LmGenes:
            class_dict.update({gene: 0})

        for pert_id in self.L[self.cell_line]:
            counter += 1
            if counter % 10 == 0:
                print('%.1f %%    \r' % (counter / length * 100), end=""),
                
            if 'dnGenes' not in self.L[self.cell_line][pert_id]:
                continue

            smiles = self.meta_smiles[self.meta_smiles['pert_id'] == pert_id]['SMILES'].values[0]
            if str(smiles) == 'nan' or str(smiles) == '-666':
                continue

            if not self.useChirality:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                canonical_smiles = smiles

            if canonical_smiles in unique_smiles or len(canonical_smiles) > 120:
                continue
            if data is not None:
                if data[data['pert_id'] == pert_id].empty:
                    continue
                else:
                    feature = data[data['pert_id'] == pert_id].drop(['pert_id'], axis=1).values[0].tolist()
            else:
                feature = self.rep.get_representation(smiles=canonical_smiles, descriptor=self.descriptor,
                                                      useChirality=self.useChirality)

            unique_smiles.append(canonical_smiles)
            dn_genes = list(set(self.L[self.cell_line][pert_id]['dnGenes']))
            class_dict = dict.fromkeys(class_dict, 0)
            for gene in dn_genes:
                if gene in class_dict:
                    class_dict.update({gene: 1})

            labels = np.fromiter(class_dict.values(), dtype=int)
            X.append(feature)
            Y.append(labels)
            perts.append(pert_id)

        x = np.asarray(X)
        y = np.asarray(Y)

        x_columns = ['SMILES']
        y_columns = list(class_dict.keys())
        if self.descriptor == 'ecfp':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_' + str(i + 1))
        elif self.descriptor == 'ecfp_autoencoder':
            for i in range(x.shape[1]-1):
                x_columns.append('ecfp_autoencoder_' + str(i + 1))
        elif self.descriptor == 'topological':
            for i in range(x.shape[1]-1):
                x_columns.append('topological_' + str(i + 1))
        elif self.descriptor == 'maccs':
            for i in range(x.shape[1]-1):
                x_columns.append('maccs_' + str(i + 1))
        elif self.descriptor == 'jtvae':
            for i in range(x.shape[1]-1):
                x_columns.append('jtvae_' + str(i + 1))
        elif self.descriptor == 'shed':
            for i in range(x.shape[1]-1):
                x_columns.append('shed_' + str(i + 1))
        elif self.descriptor == 'cats2d':
            for i in range(x.shape[1]-1):
                x_columns.append('cats2d_' + str(i + 1))

        x = pd.DataFrame(x, index=perts, columns=x_columns)
        y = pd.DataFrame(y, index=perts, columns=y_columns)
        folds = list(KFold(self.n_fold, shuffle=True, random_state=self.random_state).split(x))

        if self.random_genes:
            y_random = []
            for i in self.random_index_list:
                y_random.append(y.iloc[:, i:i + 1])
            df = y_random[0]
            for i in range(len(y_random) - 1):
                df = pd.concat([df, y_random[i + 1]], axis=1)
            y = df

        print('\nDone.')    
            
        return x, y, folds
