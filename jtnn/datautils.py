from torch.utils.data import Dataset
from mol_tree import MolTree
import numpy as np

class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        #print(len(smiles))

        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree

class PropDataset(Dataset):

    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]







#CCC(C)C(NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)CCNC(C)=O)C(=O)NC1(C)CCCC=CCCCC(C)(C(=O)NC(C)C(=O)NC(C)C(=O)NC(CC(N)=O)C(=O)NC(CCCNC(=N)N)C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(CCCNC(=N)N)C(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CCSC)NC1=O
#CC1=C2N=C(C=C3N=C(C(C)=C4C(CCC(N)=O)C(C)(CC(N)=O)C(C)(C5N=C1C(C)(CCC(=O)NCC(C)OP(=O)(O)OC1C(CO)OC(n6cnc7cc(C)c(C)cc76)C1O)C5CC(N)=O)N4[Co+]C#N)C(C)(CC(N)=O)C3CCC(N)=O)C(C)(C)C2CCC(N)=O
#C=CC(C)(O)CCC=C(CO)C(=O)OC1C(C)OC(OC(C)(C=C)CCC=C(CO)C(=O)OC2CC3(C(=O)OC4OC(CO)C(O)C(O)C4OC4OC(C)C(OC5OC(CO)C(O)C5O)C(OC5OC(CO)C(O)C(O)C5O)C4O)C(O)CC4(C)C(=CC(O)C5C6(C)CCC(OC7OC(COC8OC(C)C(O)C(O)C8OC8OCC(O)C(O)C8O)C(O)C(O)C7NC(C)=O)C(C)(C)C6CCC54C)C3CC2(C)C)C(O)C1O
#CCCCC1NC(=O)C(C)(NC(=O)C(NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)CCNC(C)=O)C(C)CC)CCCC=CCCCC(C)(C(=O)NC(C)C(=O)NC(C)C(=O)NC(CC(N)=O)C(=O)NC(CCCNC(=N)N)C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(CCCNC(=N)N)C(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC1=O
#CCC(C)N1CCC2(CC1)N=C1C(=C3NC(=O)C(C)=CC=CC(C)C(O)C(C)C(O)C(C)C(OC(C)=O)C(C)C(OC)C=COC4(C)Oc5c(C)c(O)c(c1c5C4=O)C3=O)N2
#CC(=O)OC1CC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CC(O)C5(C)C(C7=CC(=O)OC7)CCC65O)C4)OC3C)OC2C)OC(C)C1OC1OC(CO)C(O)C(O)C1O
#CC1CCCCOC(CN(C)Cc2ccc(Cl)c(Cl)c2)C(C)CN(C(C)CO)C(=O)c2cc(NC(=O)Nc3ccc4c(c3)OCO4)ccc2O1
#CC1CCCCOC(CN(C)C(=O)Cc2ccccc2)C(C)CN(C(C)CO)C(=O)c2cc(NC(=O)Nc3ccc(C(F)(F)F)cc3)ccc2O1
#CC(C)CC1C(=O)N2CCCC2C2(O)OC(NC(=O)C3C=C4c5cccc6[nH]c(Br)c(c56)CC4N(C)C3)(C(C)C)C(=O)N12
#COc1ccc(C#Cc2ccc3c(c2)C2(C(=O)N3)C(C(=O)N3CCCCCCC3)C3C(=O)OC(c4ccccc4)C(c4ccccc4)N3C2c2ccc(OCCO)cc2)cc1
#CC=CCC(C)C(O)C1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C)C(=O)NC(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(C(C)C)C(=O)N1C
#COc1ccc(CC2c3cc(OC)c(OC)cc3CC[N+]2(C)CCC(=O)OCCCCCOC(=O)CC[N+]2(C)CCc3cc(OC)c(OC)cc3C2Cc2ccc(OC)c(OC)c2)cc1OC
#CCC(C)C(NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)C(NC(=O)CCNC(C)=O)C(C)C)C(=O)NC1(C)CCCC=CCCCC(C)(C(=O)NC(C)C(=O)NC(C)C(=O)NC(CC(N)=O)C(=O)NC(Cc2ccc(O)cc2)C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(CCCNC(=N)N)C(N)=O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CCSC)NC1=O
#CCC1=CC2CN(C1)Cc1c([nH]c3ccccc13)C(C(=O)OC)(c1cc3c(cc1OC)N(C)C1C(O)(C(=O)OC)C(OC(C)=O)C4(CC)C=CCN5CCC31C54)C2
#CC1C=CC=CC=CC=CC=CC=CC=CC(OC2OC(C)C(O)C(N)C2O)CC2OC(O)(CC(O)CC(O)C(O)CCC(O)CC(O)CC(=O)OC(C)C(C)C1O)CC(O)C2C(=O)O
#C=CCNC(=O)C1C2C(=O)OC(c3ccccc3)C(c3ccccc3)N2C(c2ccccc2OCCO)C12C(=O)N(C(=O)OCCOC)c1ccc(C#CCC(C(=O)OC)C(=O)OC)cc12
#C=CC(C)(O)CCC1OCC1C(=O)OC1C(C)OC(OC(C)(C=C)CCC=C(CO)C(=O)OC2CC3(C(=O)OC4OC(CO)C(O)C(O)C4OC4OC(C)C(OC5OC(CO)C(O)C5O)C(OC5OC(CO)C(O)C(O)C5O)C4O)C(O)CC4(C)C(=CCC5C6(C)CCC(OC7OC(COC8OC(C)C(O)C(O)C8OC8OCC(O)C(O)C8O)C(O)C(O)C7NC(C)=O)C(C)(C)C6CCC54C)C3CC2(C)C)C(O)C1O
#C=Cc1c(C)c2cc3nc(cc4[nH]c(cc5nc(cc1[nH]2)C(C)=C5CCC(=O)O)c(CCC(=O)O)c4C)C1(C)C3=CC=C(C(=O)OC)C1C(=O)OC.CO
#CN1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)C(Cc4ccccc4)N3C2=O)CC2c3cccc4[nH]cc(c34)CC21.CN1CC(C(=O)NC2(C)OC3(O)C4CCCN4C(=O)C(Cc4ccccc4)N3C2=O)CC2c3cccc4[nH]cc(c34)CC21
#CCCCC(NC(=O)C(CCC(N)=O)NC(=O)C(NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)CCNC(C)=O)C(C)CC)C(=O)NC(CC(C)C)C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(C)C(=O)NC(C)C(=O)NC(CC(N)=O)C(=O)NC(CCCNC(=N)N)C(=O)NC1(C)CCCC=CCCCC(C)(C(=O)NC(CCCNC(=N)N)C(N)=O)NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)C(CCC(=O)O)NC1=O
#CC1CCC2(C(=O)OC3OC(COC4OC(CO)C(OC5OC(C)C(O)C(O)C5O)C(O)C4O)C(O)C(O)C3O)CCC3(C)C(=CCC4C5(C)CC(O)C(O)C(C)(CO)C5CCC43C)C2C1C
#CC=Cc1ccc2n(c1=O)CC1C(CO)C(C(=O)NC3Cc4ccccc4C3)N(CC)C21.CC=Cc1ccc2n(c1=O)CC1C(CO)C(C(=O)NC3Cc4ccccc4C3)N(CC)C21
#CC=Cc1ccc2n(c1=O)CC1C(CO)C(C(=O)N(C)C)N(CCC(F)(F)F)C21.CC=Cc1ccc2n(c1=O)CC1C(CO)C(C(=O)N(C)C)N(CCC(F)(F)F)C21
#CC1OC(OC2CCC3(C)C(CCC4(C)C3C=CC35OCC6(CCC(C)(C)CC63)C(O)CC45C)C2(C)CO)C(OC2OC(CO)C(O)C(O)C2O)C(OC2OC(CO)C(OC3OC(CO)C(O)C(O)C3O)C(O)C2O)C1O
#CCC(C)C(N)C1=NC(C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(C(=O)NC2CCCCNC(=O)C(CC(N)=O)NC(=O)C(CC(=O)O)NC(=O)C(Cc3cnc[nH]3)NC(=O)C(Cc3ccccc3)NC(=O)C(C(C)CC)NC(=O)C(CCCN)NC2=O)C(C)CC)CS1
#CC(N)C(=O)NCC(=O)NC1CSSCC(C(=O)O)NC(=O)C(CO)NC(=O)C(C(C)O)NC(=O)C(Cc2ccccc2)NC(=O)C(C(C)O)NC(=O)C(CCCCN)NC(=O)C(Cc2c[nH]c3ccccc23)NC(=O)C(Cc2ccccc2)NC(=O)C(Cc2ccccc2)NC(=O)C(CC(N)=O)NC(=O)C(CCCCN)NC1=O
#COC(C(=O)C(O)C(C)O)C1Cc2cc3cc(OC4CC(OC5CC(O)C(OC)C(C)O5)C(OC(C)=O)C(C)O4)c(C)c(O)c3c(O)c2C(=O)C1OC1CC(OC2CC(OC3CC(C)(O)C(C)C(C)O3)C(O)C(C)O2)C(O)C(C)O1
#Cc1c2oc3c(C)ccc(C(=O)OC(C)C4NC(=O)C(C(C)C)N(C)C(=O)CN(C)C(=O)C5CCCN5C(=O)C(C(C)C)NC4=O)c3nc-2c(C(=O)OC(C)C2NC(=O)C(C(C)C)N(C)C(=O)CN(C)C(=O)C3CCCN3C(=O)C(C(C)C)NC2=O)c(N)c1=O
#CCCCC1NC(=O)C(C)(NC(=O)C(NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)C(CCCNC(=N)N)NC(=O)C(NC(=O)CCNC(C)=O)C(C)C)C(C)CC)CCCC=CCCCC(C)(C(=O)NC(C)C(=O)NC(C)C(=O)NC(CC(N)=O)C(=O)NC(Cc2ccc(O)cc2)C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(CCCNC(=N)N)C(N)=O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC1=O
#C=C(NC(=O)C(=C)NC(=O)c1csc(C2=NC3c4csc(n4)C4NC(=O)c5csc(n5)C(C(C)(O)C(C)O)NC(=O)C5CSC(=N5)C(=CC)NC(=O)C(C(C)O)NC(=O)c5csc(n5)C3(CC2)NC(=O)C(C)NC(=O)C(=C)NC(=O)C(C)NC(=O)C(C(C)CC)NC2C=Cc3c(C(C)O)cc(nc3C2O)C(=O)OC4C)n1)C(N)=O
#C=CC(C)(O)CCC=C(C)C(=O)OC1C(C)OC(OC(C)(C=C)CCC=C(CO)C(=O)OC2CC3(C(=O)OC4OC(CO)C(O)C(O)C4OC4OC(C)C(OC5OC(CO)C(O)C5O)C(OC5OC(CO)C(O)C(O)C5O)C4O)C(O)CC4(C)C(=CCC5C6(C)CCC(OC7OC(COC8OC(C)C(O)C(O)C8OC8OCC(O)C(O)C8O)C(O)C(O)C7NC(C)=O)C(C)(C)C6CCC54C)C3CC2(C)C)C(O)C1O

