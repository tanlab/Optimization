#!/usr/bin/env python
# coding: utf-8

# In[3]:


import optuna
from keras.models import load_model


# In[4]:


import numpy as np


# In[5]:


up_model = load_model("trained_models/PC3_multi_task_model_up.h5")
dn_model = load_model("trained_models/PC3_multi_task_model_dn.h5")


# In[6]:


from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score

gm_scorer = make_scorer(geometric_mean_score, greater_is_better=True, average='binary')


# In[ ]:


def objective(trial):
    test_pred = []
    for i in range(56):
        name = 'jtvae_' + str(i)
        test_pred.append(trial.suggest_uniform(name, 0, 1))
#     print(test_pred)
    up_pred = up_model.predict(np.asarray(test_pred).reshape(1,-1))
    dn_pred = dn_model.predict(np.asarray(test_pred).reshape(1,-1))
#     print(up_pred)
    up_pred_bin = []
    dn_pred_bin = []
    
    for i in range(len(up_pred)):
        up_pred_bin.append(round(up_pred[i][0][0]))
    for i in range(len(dn_pred)):
        dn_pred_bin.append(round(dn_pred[i][0][0]))
    
#     print(len(up_pred_bin))
    '''
    Magic happens
    '''
    up_score = geometric_mean_score(up_genes_228, up_pred_bin)
    dn_score = geometric_mean_score(dn_genes_228, dn_pred_bin)
#     print(up_score)
    return((up_score + dn_score) / 2)


# In[ ]:


import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


# In[ ]:


up_harmonizome = pd.read_csv('harmonizome_diseases/harmonizome_dn_binarized_use_for_up_model.csv')
dn_harmonizome = pd.read_csv('harmonizome_diseases/harmonizome_up_binarized_use_for_dn_model.csv')


# In[ ]:


gene_names = pd.read_csv('100_gene_names/meta_Probes_info.csv',index_col='probe')


# In[ ]:



f = open("100_gene_names/PC3_multi_task_gene_list_up.txt", "rt")
mcf7_up_genes = f.read()
mcf7_up_genes = mcf7_up_genes[:-1]
f.close()
print(mcf7_up_genes.split('\n'))


# In[ ]:



f = open("100_gene_names/PC3_multi_task_gene_list_dn.txt", "rt")
mcf7_dn_genes = f.read()
mcf7_dn_genes = mcf7_dn_genes[:-1]
f.close()
print(mcf7_dn_genes.split('\n'))


# In[ ]:


up_genes_228 = []
for gene in up_harmonizome.iloc[217].index[2:]:
    if(gene_names.loc[gene][0] in mcf7_up_genes.split('\n')):
        up_genes_228.append(up_harmonizome.iloc[217][gene])


# In[ ]:


dn_genes_228 = []
for gene in dn_harmonizome.iloc[217].index[2:]:
    if(gene_names.loc[gene][0] in mcf7_dn_genes.split('\n')):
        dn_genes_228.append(dn_harmonizome.iloc[217][gene])


# In[ ]:


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100000)
    print(study.best_trial)


# In[14]:


import os
os.environ['USE_CPU']


# In[28]:


# from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
# from keras.models import Model
# from keras.optimizers import SGD
# from keras.models import load_model
# from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import json
import pandas as pd
import numpy as np
import torch
import sys
import copy
# import tensorflow as tf
from jtnn import *

sys.path.append('./jtnn')
vocab = [x.strip("\r\n ") for x in open("unique_canonical_train_vocab.txt")]
vocab = Vocab(vocab)

hidden_size = 450
latent_size = 56
depth = 3
stereo = True

model_jtvae = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model_jtvae.cuda()
model_jtvae.load_state_dict(torch.load("Models/model.iter-9-6000", map_location=torch.device('cuda')))  # opts.model_path


optimize_edilen_smi = model_jtvae.reconstruct2(torch.from_numpy(np.asarray([acc_opt[0:28]])).float().cuda(),
                                               torch.from_numpy(np.asarray([acc_opt[28:56]])).float().cuda())
optimize_edilen_smi


# In[10]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


acc_opt =  [0.21333110154807555,  0.5696635663553961,  0.041857031476346424,  0.5560664591620894,  0.028210498166085233,  0.7445936092457618,  0.4824778156993952,  0.09311065395660822,  0.2925614208921216,  0.019231491250824812,  0.4015278849183026,  0.7097219306730131,  0.30521377607517114,  0.201376261554047,  0.09161383295204663,  0.3026530370458903,  0.8250423952436019,  0.910108740133988,  0.01828063072619396,  0.05659609105010166,  0.7954110984694707,  0.5432823792530542,  0.1596515990093705,  0.14630748414687136,  0.6104404614002065,  0.2110650248136642,  0.7798225395434372,  0.10335836827442857,  0.6888627939272463,  0.650712914764352,  0.32021750962790174,  0.5940325004684724,  0.766149854726883,  0.00888778216811122,  0.6318963245236938,  0.1614476362351259,  0.3548044251269435,  0.6567831561767696,  0.8792052872809961,  0.46907401201427834,  0.7447526787697938,  0.6032635624747409,  0.7341386168129296,  0.6074866706543967,  0.9016413095009643,  0.1878630142560771,  0.6576493824969326,  0.5845817142726869,  0.773655675462739,  0.03914063460894965,  0.07365230599661123,  0.7475142412994042,  0.24358529812824645,  0.8356452953827569,  0.7588869253061779,  0.6051541318429876]


# DKit ERROR: [17:54:16] Explicit valence for atom # 1 C, 6, is greater than permitted
# 
# 'O=C(C[NH+]1CCCC1)NCc1ccc2c3c(oc2c1)CCCCC3'

# In[18]:


acc_opt = [ 5.498360324151777,  -7.283728411540407,  0.2878314819440486,  3.014250620522354,  -0.967431279452156,  4.641302984740675,  -3.1588656820109704,  3.972458648433881,  -9.413742843642183,  -0.7216084318552025,  7.402066499743269,  5.317071850267407,  6.092359654651686,  1.8436069707358793,  -2.619408235547732,  -3.197497600275462,  8.583440507119473,  -3.635466095673355,  -3.1976967668422676,  -1.8238226707522562,  -6.714634059209021,  -1.2660136277351348,  3.0922139425544417,  2.6175807949924828,  -4.69718520869906,  -8.986576972087992,  -3.505895499627265,  -8.43571370370084,  2.978047901979137,  7.616661574744671,  -3.045310272073862,  3.580540802989734,  -3.8157569448261914,  -7.680722248177964,  -2.63336795521246,  -4.712131231298505,  -3.282667271075425,  -6.552422541975782,  6.80660949637101,  -1.5622485975922995,  -8.220184839069104,  6.107353992241136,  0.6734481121657852,  -3.3713570792288405,  -3.2427643059993736,  3.883805953914077,  -8.336208000948284,  -9.911301611222274,  -2.497505788438981,  -3.9419343943398237,  5.542249664472779,  -9.654161434799283,  -4.733527232143906,  -2.4831274594172195,  4.231148317485828,  0.010494957601254296]


# DKit ERROR: [17:56:42] Explicit valence for atom # 1 O, 4, is greater than permitted
# 
# 'Nc1cnccc1CC(N)C1OCCC1=O'

# In[19]:


acc_opt = [ 0.3270286030948731,  0.7769213794645228,  0.39073894666199616,  0.07760645155596117,  0.37119911994152655,  0.18778274083011043,  0.47955389175258833,  0.1440611941733854,  0.018771554764599786,  0.8878619265615294,  0.36875872806097815,  0.9198155942935811,  0.9857755434254801,  0.18573085182523796,  0.06522665006901446,  0.9657623837427676,  0.9243253259568084,  0.06545777393408632,  0.9985347842573016,  0.23158297375740622,  0.7144680604238436,  0.7391293590614246,  0.8786050729907185,  0.9019306292889296,  0.585286408569106,  0.07443852738232454,  0.07992074612419975,  0.2828196051326124,  0.28154648867308274,  0.0850421466544208,  0.17495363663701446,  0.9512828110070497,  0.8606237722379475,  0.255084690083291,  0.3445359211071884,  0.22418045908080064,  0.45640402348738457,  0.5789210298880988,  0.6240532934601511,  0.3510174537371844,  0.19752508528984192,  0.7325866224706316,  0.9273919364118667,  0.45041685269879606,  0.051711278548190416,  0.9624543017427802,  0.9305463533996738,  0.5129820665442999,  0.9801519901365725,  0.3324563363645149,  0.18144736514382862,  0.460360172184675,  0.7573817361183744,  0.07990060315161476,  0.8729904781139662,  0.8300648995006007]


# O=C(C[NH+]1CCC1)Nc1ccccc1C1C=CCCC1

# In[23]:


acc_opt = [ 0.7686496397917053,  0.7290275878571697,  0.8962781501719216,  0.001828737547023557,  0.1048716235507492,  0.8011520180890929,  0.6306522099118735,  0.3188711551861658,  0.08192035925625778,  0.3091039767255279,  0.33422460907244705,  0.8298325103566517,  0.0008548863002470607,  0.07324644470821258,  0.03828112648197212,  0.4075562411799588,  0.9987551636470469,  0.26435339566439614,  0.9344262405572779,  0.5004984544338399,  0.42454752877625057,  0.45267487475018936,  0.7030695625817183,  0.8402649239317417,  0.6678831615234574,  0.10395650421375778,  0.5199166544223349,  0.8030870790264286,  0.9680949777512775,  0.5135365089851679,  0.509097199487812,  0.023608505282887042,  0.9493246829412839,  0.06090915419549155,  0.5723343639373466,  0.4413230272923972,  0.9182666337128564,  0.36456617790164264,  0.7542928025744077,  0.8354755377135046,  0.9279859804414375,  0.960396829279954,  0.09929569736225202,  0.9419639737338528,  0.17188382899882487,  0.036799425804540674,  0.1157430976001381,  0.24592435191692638,  0.4698848937698159,  0.5979878887629461,  0.36642563904377123,  0.6741545622114725,  0.9988902101975098,  0.9982026728076733,  0.10768032725501586,  0.9228300189448886]


# 'O=C(CN1CCOCC1)Oc1ccc(C2NCNC2=S)cc1'

# In[25]:


acc_opt =[ -0.858197347915798,  0.3524764731329588,  -0.9978473863605177,  -0.9985490482442735,  0.9561180606342758,  0.0455756076738017,  -0.35260710331880296,  -0.23043932143940313,  0.013160319062403586,  0.8200197821820497,  -0.7050092082435317,  0.27991601303173164,  -0.5816223891828064,  -0.9961988332774885,  0.08097522318898452,  0.34361723351713747,  0.7325220432044199,  -0.3419073459162655,  0.849320931270694,  0.31573819395245795,  -0.6236646726605716,  -0.39869591227221335,  -0.25826189374948305,  0.5883322926166812,  -0.5761599608732091,  -0.9992434978406926,  -0.4354650822148419,  -0.51208542110699,  0.5398753989319423,  -0.8377080637467194,  -0.2506876746600207,  0.44621274470271355,  -0.3777213161514557,  -0.17663565892775454,  0.368826471952738,  0.09023891247047007,  0.394854430514563,  0.580289642113775,  -0.9984596177446557,  -0.8916626780503483,  -0.7771840284620944,  -0.13766998795415228,  0.40025657357523353,  -0.16437617526654583,  -0.9254559856245337,  0.3475873076769103,  -0.8490270666816755,  0.6963786162019766,  -0.7024158699684337,  0.2981148911204836,  0.3402626852861166,  -0.08501865418235921,  0.8242520985809543,  -0.5157672377693626,  -0.8734649902000446,  0.6233965420564741]


# 'Cc1cncc(NC(=O)Cc2cccc(O)c2)c1'

# In[27]:


acc_opt =   -0.7806234115315263,  -0.38833015388856235,  -0.8163337350716091,  -0.8493217427623899,  0.7830709225752265,  -0.12288049219115854,  -0.19318288964555255,  -0.04538600283713162,  -0.25803767253279086,  0.9041515764705863,  -0.042678251403517,  0.3928457195027314,  -0.1405339430186077,  -0.8793191925383338,  0.00485630116883843,  0.6756134390501861,  0.8981140264603061,  -0.23350940602912176,  0.6512578238686421,  -0.5018438826149468,  -0.8197555214256255,  -0.6530483073268175,  -0.7707785133997186,  0.6247644048216368,  -0.8635636858218838,  -0.5602986102859554,  0.19558405765092157,  -0.5383909358425416,  0.48898024724112904,  -0.8591051610252629,  -0.0447118301811973,  0.6434611799960981,  -0.4275652356615174,  0.01844703104041004,  0.7243954884492632,  -0.30379761727232407,  0.273244910095718,  0.9608458540548884,  -0.8562454091783177,  -0.7845556826073111,  -0.8302617475169067,  -0.4608740110580266,  0.12194401513211976,  0.214115647508077,  -0.9433837966824652,  -0.1410347141234244,  -0.5894546712047176,  0.9248418376769593,  -0.9938634745963044,  -0.21930777357063427,  0.2373173659223085,  0.39741735158207425,  0.3763387077674806,  -0.5182235075865299,  -0.7875522923825826,  0.5954277504865219


# 'Cc1cncc(NC(=O)Cc2ccc(O)cc2)c1'

# In[21]:


up_pred = up_model.predict(np.asarray(acc_opt).reshape(1,-1))
dn_pred = dn_model.predict(np.asarray(acc_opt).reshape(1,-1))
#     print(up_pred)
up_pred_bin = []
dn_pred_bin = []

for i in range(len(up_pred)):
    up_pred_bin.append(round(up_pred[i][0][0]))
for i in range(len(dn_pred)):
    dn_pred_bin.append(round(dn_pred[i][0][0]))
