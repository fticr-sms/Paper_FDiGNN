# standard library imports
# import os
import pickle
import json
import sys
import time
# from fractions import Fraction

# scientific python imports
import pandas as pd
import numpy as np
import seaborn as sns

# import sklearn.ensemble as skensemble
# from sklearn.metrics import (roc_auc_score, roc_curve, auc,
#                              f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import networkx as nx
# import xgboost as xgb

# metabolinks and "in folder" modules
import metabolinks.transformations as transf
# venn.py file  (has to be in the same folder)
# metanalysis_standard.py file (has to be in the same folder)
import metanalysis_standard as metsta

# form_assign_func.py file (has to be in the same folder)
# import form_assign_func as form_afunc
import MDiN_functions as md

from tqdm import tqdm

import torch
from torch.nn import Linear, Softmax, Softmax2d, Sequential, BatchNorm1d, ReLU, Dropout, LeakyReLU, PReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATv2Conv, TAGConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.pool import TopKPooling

from torch_geometric.nn.pool import SAGPooling
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

print("All imports concluded")

# Report versions
print("PyTorch version", torch.__version__)
print("CUDA version", torch.version.cuda)

#### If a gpu is available (if it is not, comment the 3 lines below)
#torch.cuda.current_device()
torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'\nUSING DEVICE {device}\n')

#### If a gpu is not available (if it is, comment the line below)
# device = 'cpu'


start_time = time.time()

with open('Data/LID_GNNPathwayTest_graphlets_size5.txt') as a:
    gs = a.read().split('\n')

with open('Data/LID_GNNPathwayTest_singlenodes_size5.txt') as a:
    sns_p = a.read().split('\n')

graphlets = []
for g in gs:
    graphlets.append(g.split(', '))

single_nodes = []
for g in sns_p:
    single_nodes.append(g.split(', '))

graphlets = graphlets[:-1]
single_nodes = single_nodes[:-1]

pathway_to_alter = graphlets[int(sys.argv[-1])]
single_nodes = single_nodes[int(sys.argv[-1])]

print(f'Running Graphlets {sys.argv[-1]}. {pathway_to_alter} / {single_nodes}')

# Filename for the data to import
filename_TreatedData = 'Data/LID_AllTreatedData_Final.xlsx'
filename_proc = 'Data/LID_ProcData_Final.pickle'
filename_treat = 'Data/LID_TreatedData_Final.pickle'
target_name = 'Data/LID_Target_Final.txt'

bin_data = pd.read_excel(filename_TreatedData, sheet_name='BinSim Treated Data')#.set_index('Unnamed: 0').T
bin_data = bin_data.set_index('Bucket label').T

univariate_data = pd.read_excel(filename_TreatedData, sheet_name='MVI+Norm Data')#.set_index('Unnamed: 0')

processed_data = pd.read_pickle(filename_proc)
treated_data = pd.read_pickle(filename_treat).T
bin_data.columns = treated_data.columns

with open(target_name) as a:
    tg = a.readlines()
target = [t.strip() for t in tg]
sample_cols = list(treated_data.index)

np.random.seed(301430)
aleatorized_treated_data = treated_data.T.copy()
aleatorized_bin_data = bin_data.T.copy()
for i in aleatorized_treated_data.index:
    n = np.random.randint(100000000)
    aleatorized_treated_data.loc[i] = aleatorized_treated_data.loc[i].sample(frac=1, random_state=n).values
    aleatorized_bin_data.loc[i] = aleatorized_bin_data.loc[i].sample(frac=1, random_state=n).values
aleatorized_treated_data = aleatorized_treated_data.T#.iloc[0].value_counts().sort_index()
aleatorized_bin_data = aleatorized_bin_data.T

np.random.seed(40012*(int(sys.argv[-1])+1))
changed_treated_data = treated_data.copy()
changed_aleatorized_treated_data = aleatorized_treated_data.copy()
changed_bin_data = bin_data.copy()
changed_aleatorized_bin_data = aleatorized_bin_data.copy()

sample_dict = dict(zip(sample_cols, target))
cp_samples = [i for i in sample_dict.keys() if sample_dict[i] == 'start']
rep_samples = [i for i in sample_dict.keys() if sample_dict[i] == 'end']

all_int_values = treated_data.values.flatten()

low_int_values = all_int_values[all_int_values > np.quantile(all_int_values, 0.03)]
low_int_values = low_int_values[low_int_values < np.quantile(all_int_values, 0.1)]

high_int_values = all_int_values[all_int_values > np.quantile(all_int_values, 0.9)]
high_int_values = high_int_values[high_int_values < np.quantile(all_int_values, 0.97)]

for node in pathway_to_alter + single_nodes:
    pos_values = np.random.choice(high_int_values, 177, replace=False)
    changed_treated_data.loc[cp_samples, node] = pos_values
    changed_aleatorized_treated_data.loc[cp_samples, node] = pos_values
    changed_bin_data.loc[cp_samples, node] = [1,] * 177
    changed_aleatorized_bin_data.loc[cp_samples, node] = [1,] * 177

    neg_values = np.random.choice(low_int_values, 177, replace=False)
    changed_treated_data.loc[rep_samples, node] = neg_values
    changed_aleatorized_treated_data.loc[rep_samples, node] = neg_values
    changed_bin_data.loc[rep_samples, node] = [0,] * 177
    changed_aleatorized_bin_data.loc[rep_samples, node] = [0,] * 177


datasets = {
    'Normal + Path': changed_treated_data,
    'Aleatorized + Path': changed_aleatorized_treated_data,
}
datasets_bin = {
    'Normal + Path': changed_bin_data,
    'Aleatorized + Path': changed_aleatorized_bin_data,
}

n_fold = 5
iter_num = 10

# Choose a number for the seed for consistent results
np.random.seed(65824802)

n_trees=200 # Number of trees in the model

RF_accus = {}
RF_imp_feats = {}

for key in datasets:

    RF_results = metsta.RF_model(datasets[key], target, regres=False, # Data, labels and if it's a regression or classification
                    return_cv=True, iter_num=iter_num, # If you want cross validation results and number of iterations for it
                    n_trees=n_trees, # Number of trees in the model
                    cv=None, n_fold=n_fold, random_state=174, # Choose a method of cross-validation (None is stratified cv) and the number of folds
    
             metrics = ('accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted')) # Choose the performance metrics
    RF_accus[key] = np.mean(RF_results['accuracy'])
    RF_imp_feats[key] = RF_results['imp_feat']

imp_feats_rf = processed_data[['Probable m/z']].copy()
imp_feats_rf.insert(0,'Bucket label', imp_feats_rf.index)
for key in datasets:
    imp_feats_rf.insert(1,key, '')
    for n in range(len(RF_imp_feats[key])):
        imp_feats_rf[key].loc[datasets[key].columns[RF_imp_feats[key][n][0]]] = RF_imp_feats[key][n][1]
rank_imp_feats_rf = imp_feats_rf.rank(ascending=False)


# above is to supress PLS warnings
# Choose a number for the seed for consistent results
np.random.seed(65824802)

n_comp = 6 # Number of components of PLS-DA model - very important

PLSDA_accus = {}
PLSDA_imp_feats = {}

for key in datasets:

    PLSDA_results = metsta.PLSDA_model_CV(datasets[key], target, regres=False, # Data, target and if it's a regression
                       n_comp=n_comp, # Number of components of PLS-DA model - very important
                       kf=None, n_fold=n_fold, # Cross validation to use (none is stratified CV) and nº of folds
                       iter_num=iter_num, # Number of iterations of cross-validation to do
                       encode2as1vector=True,
                       scale=False, # Set scale to True only if you did not do scaling in pre-treatments
                       feat_type='VIP',
                    random_state=174) # Feature Importance Metric to use, default is VIP scores (see function for others)

    PLSDA_accus[key] = np.mean(PLSDA_results['accuracy'])
    PLSDA_imp_feats[key] = PLSDA_results['imp_feat']

imp_feats_plsda = processed_data[['Probable m/z']].copy()
imp_feats_plsda.insert(0,'Bucket label', imp_feats_plsda.index)
for key in datasets:
    imp_feats_plsda.insert(1,key, '')
    for n in range(len(PLSDA_imp_feats[key])):
        imp_feats_plsda[key].loc[datasets[key].columns[PLSDA_imp_feats[key][n][0]]] = PLSDA_imp_feats[key][n][1]
rank_imp_feats_plsda = imp_feats_plsda.rank(ascending=False)


temp_df = processed_data.copy()

for i in temp_df.index:
    fs = temp_df.loc[i, 'Matched HMDB formulas']
    if type(fs) == list:
        fs = list(set(fs))
        if len(fs) == 1:
            temp_df.loc[i, 'Formula_Assignment'] = fs[0]
            temp_df.loc[i, 'Formula_Assignment Adduct'] = temp_df.loc[i, 'Matched HMDB formulas'][0]
        else:
            counted = False
            for f in fs:
                if f == temp_df.loc[i, 'Formula_Assignment']:
                    counted = True
            if counted == False:
                new_f = []
                for f in fs:
                    a = md.formula_process(f)
                    if a['C'] != 0 and a['H'] != 0:
                        if len(a) == 8:
                            if a['Cl'] == 0 and a['F'] == 0:
                                new_f.append(f)
                if len(new_f) == 1:
                    temp_df.loc[i, 'Formula_Assignment'] = new_f[0]
                else:
                    if len(new_f) > 1:
                        if 'C12H22O11' in new_f:
                            temp_df.loc[i, 'Formula_Assignment'] = 'C12H22O11'
                        else:
                            print('---')
                            print(len(new_f))
                            print(new_f)
                            print(temp_df.loc[i, 'Formula_Assignment'])
                            print(i)
                            print('---------------')
                    else:
                        print(fs)
                        print(temp_df.loc[i, 'Formula_Assignment'])
                        print('---------------')


formula_df = temp_df
# Get the formulas from formula assignment, excluding isotopes
formula_df = formula_df.dropna(subset='Formula_Assignment')
formula_df = formula_df.loc[[i for i in formula_df.index if 'iso.' not in formula_df.loc[i, 'Formula_Assignment']]]
# Add the counts of the different elements in columns
elems = metsta.create_element_counts(formula_df, formula_subset=['Formula_Assignment',], compute_ratios=False, drop_duplicates=False)
filt_elems = elems.iloc[:,:-1]

# Create MDB list of accepted chemical transformations
MDB = ['H2','CH2','CO2','O','CH2O','NCH','O(N-H-)','S','CONH','PO3H','NH3(O-)','SO3','CO', 'C2H2O', 'H2O']
results = {}
for i in MDB:
    results[i] = md.formula_process(i, elems=filt_elems.columns)
MDB_df = pd.DataFrame(results).T

with open('SMPDB_MetaNetwork_general.pickle', 'rb') as f:
    FDiN_knowledge = pickle.load(f)
node_list = list(FDiN_knowledge.nodes())

#  Restrict Information to reduce memory usage
for u, v, data in FDiN_knowledge.edges(data=True):
    for name in ['Pathways', 'SMPDB_IDs']:
        data.pop(name)
for u, data in FDiN_knowledge.nodes(data=True):
    for name in ['Names', 'Compound', 'SMPDB_IDs', 'HMDB_ID']:
        data.pop(name)


# See which of these formulas were detected in our dataset
keep_idxs = []
keep_formulas = []
keep_pathways = []
form_to_idx = {}

for i in temp_df.index:
    counted = False
    fs = temp_df.loc[i, 'Matched HMDB formulas']
    if type(fs) == list:
        fs = list(set(fs))
        if len(fs) == 1:
            form = fs[0]
            if form in node_list:
                keep_idxs.append(i)
                keep_formulas.append(form)
                if form in form_to_idx:
                    form_to_idx[form].append(i)
                else:
                    form_to_idx[form] = [i,]
                    keep_pathways.extend(FDiN_knowledge.nodes()[form]['Pathways'])
                counted = True
        else:
            form_in_node_list = []
            for f in fs:
                if f in node_list:
                    form_in_node_list.append(f)
            if len(form_in_node_list) >= 1:
                for f in form_in_node_list:
                    keep_idxs.append(i)
                    keep_formulas.append(f)
                    if f in form_to_idx:
                        form_to_idx[f].append(i)
                    else:
                        form_to_idx[f] = [i,]
                        keep_pathways.extend(FDiN_knowledge.nodes()[f]['Pathways'])
                counted = True

    if not counted:
        fs = temp_df.loc[i, 'Formula_Assignment']
        if type(fs) == str:
            if fs in node_list:
                keep_idxs.append(i)
                keep_formulas.append(fs)
                if fs in form_to_idx:
                    form_to_idx[fs].append(i)
                else:
                    form_to_idx[fs] = [i,]
                    keep_pathways.extend(FDiN_knowledge.nodes()[fs]['Pathways'])

# Subgraph the FDiN to only keep these formulas as information for FDiGNN
FDiN_knowledge = FDiN_knowledge.subgraph(keep_formulas)


# FDiN basis
FDiN = nx.Graph()
FDiN.add_nodes_from(filt_elems.index) # Each formula is a node
# Adding relevant attributes
nx.set_node_attributes(FDiN, formula_df['Formula_Assignment'].to_dict(), name='Formula')

# Adding simple edges
for formula in filt_elems.index:
    poss_formulas = filt_elems.loc[formula] + MDB_df
    for i in poss_formulas.index:
        poss_matches = filt_elems[(filt_elems == poss_formulas.loc[i]).sum(axis=1) == len(MDB_df.columns)]
        for node in poss_matches.index:
            FDiN.add_edge(formula, node, Transformation=i, Weight=1)

# Adding Knowledge-based edges from the metabolic-knowledge based network
for n1 in FDiN.nodes():
    formA = FDiN.nodes()[n1]['Formula']
    if formA in FDiN_knowledge.nodes():
        for n2 in FDiN.nodes():
            formB = FDiN.nodes()[n2]['Formula']
            if formA != formB:
                if formB in FDiN_knowledge.nodes():
                    if (formA, formB) in FDiN_knowledge.edges():
                        if (n1, n2) in FDiN.edges():
                            FDiN.edges()[(n1, n2)]['Weight'] = 2
                        else:
                            FDiN.add_edge(n1, n2, Transformation='Knowledge', Weight=2)
                    elif (formB, formA) in FDiN_knowledge.edges():
                        print('-------')
print(f'Total with: {len(FDiN.edges())} edges.')

comps = []
for i in sorted(nx.connected_components(FDiN), key=len, reverse=True):
    if len(i) > 8:
        comps.extend(i)
FDiN = FDiN.subgraph(comps)
print(f'Filtered FDiN with: {len(FDiN.edges())} edges.')


sFDiNs_full = {}
for key in datasets:
    sFDiNs_full[key] = {}
    for samp in sample_cols:
    
        sFDiNs_full[key][samp] = FDiN.copy()
        ints = {i: datasets[key].loc[samp, i] for i in formula_df.index}
        pres = {i: datasets_bin[key].loc[samp, i] for i in formula_df.index}
        # Storing intensity of feature in sample, mass and node degree on the nodes
        intensity_attr = dict.fromkeys(sFDiNs_full[key][samp].nodes(),0)
        for m in sFDiNs_full[key][samp].nodes():
            intensity_attr[m] = {'mass':formula_df.loc[m,'Probable m/z']/100, 'intensity': ints[m], 'presence':pres[m]}
        nx.set_node_attributes(sFDiNs_full[key][samp], intensity_attr)


# Node Features
node_attrs = list(sFDiNs_full[key][samp].nodes()[list(sFDiNs_full[key][samp].nodes())[0]].keys())[1:]

# Edge Attributes
edge_attrs = list(sFDiNs_full[key][samp].edges()[list(sFDiNs_full[key][samp].edges())[0]].keys())
edge_attrs.remove('Transformation')


all_data_full = {}

for key in datasets:

    # Convert the sFDiNs into PyTorch geometric
    data_list_full = []
    for samp in sFDiNs_full[key]:
        pyg_graph = from_networkx(sFDiNs_full[key][samp],
                                  group_node_attrs=list(sFDiNs_full[key][samp].nodes()[list(sFDiNs_full[key][samp].nodes())[0]].keys())[1:],
                                  group_edge_attrs=edge_attrs)
        data_list_full.append(pyg_graph.to(device))
    
    # Adding target information to the sFDiNs
    for g in range(len(target)):
        if target[g] == 'end':
            data_list_full[g].y = torch.FloatTensor([1]).type(torch.LongTensor).to(device)
        else:
            data_list_full[g].y = torch.FloatTensor([0]).type(torch.LongTensor).to(device)
    all_data_full[key] = data_list_full


# Setting up the model
class FDiGNN_TAG(torch.nn.Module):
    def __init__(self, hidden_channels, drop, n_node_feat, K):
        super(FDiGNN_TAG, self).__init__()
        torch.manual_seed(89356)
        self.conv1 = TAGConv(n_node_feat, hidden_channels, K=K)
        self.norm1 = BatchNorm1d(hidden_channels)
        self.conv2 = TAGConv(hidden_channels, hidden_channels, K=K)
        self.norm2 = BatchNorm1d(hidden_channels)
        self.conv3 = TAGConv(hidden_channels, hidden_channels, K=K)
        self.norm3 = BatchNorm1d(hidden_channels)
        self.conv4 = TAGConv(hidden_channels, hidden_channels, K=K)
        self.norm4 = BatchNorm1d(hidden_channels)
        self.pooling = GlobalAttentionPooling(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
        self.drop = drop
        self.last_att_conv1 = None
        self.last_att_conv2 = None
        self.last_att_conv3 = None
        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()
        self.leakyrelu3 = nn.LeakyReLU()
        self.leakyrelu4 = nn.LeakyReLU()

    def forward(self, x, edge_index, batch, edge_weight, retrieve_steps=False):
        # 1. Obtain node embeddings 
        x1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        x1_relu = self.leakyrelu1(x1)
        x1_norm = self.norm1(x1_relu)
        x1_drop = F.dropout(x1_norm, p=self.drop, training=self.training)
        x2 = self.conv2(x1_drop, edge_index, edge_weight=edge_weight)
        x2_relu = self.leakyrelu2(x2)
        x2_norm = self.norm2(x2_relu)
        x2_drop = F.dropout(x2_norm, p=self.drop, training=self.training)
        x3 = self.conv3(x2_drop, edge_index, edge_weight=edge_weight)
        x3_relu = self.leakyrelu3(x3)
        x3_norm = self.norm3(x3_relu)
        x3_drop = F.dropout(x3_norm, p=self.drop, training=self.training)
        x4 = self.conv4(x3_drop, edge_index, edge_weight=edge_weight)
        x4_relu = self.leakyrelu4(x4)
        x4_norm = self.norm4(x4_relu)
        x4_drop = F.dropout(x4_norm, p=self.drop, training=self.training)

        # 2. Readout layer
        x_emb = self.pooling(x4_drop, batch)

        # 3. Apply a final classifier
        x_emb = F.dropout(x_emb, p=self.drop, training=self.training)
        x_emb = self.lin1(x_emb)
        x_emb = self.lin2(x_emb)
        if retrieve_steps:
            self.x = x
            self.x1 = x1
            self.x1_relu = x1_relu
            self.x1_norm = x1_norm
            self.x1_drop = x1_drop
            self.x2 = x2
            self.x2_relu = x2_relu
            self.x2_norm = x2_norm
            self.x2_drop = x2_drop
        return x_emb

class GlobalAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentionPooling, self).__init__()
        self.attention_nn = nn.Sequential(nn.Linear(in_channels, 1), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()
        self.last_scores = None
        self.x_weighted = None
        
    def forward(self, x, batch):
        scores = self.attention_nn(x).squeeze(-1)
        scores = softmax(scores, batch)
        x_weighted = x * scores.unsqueeze(-1)
        self.last_scores = scores
        self.x_weighted = x_weighted
        graph_embedding = scatter_add(x_weighted, batch, dim=0)
        
        return graph_embedding

    def get_attention_scores(self):
        return self.last_scores, self.x_weighted

def train(model, train_loader, optimizer):
    model.train()
    losses = []
    grad_norms = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float())  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        losses.append(loss.to('cpu').detach().numpy())
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return np.mean(losses), grad_norms, model

def test(model, loader):
    model.eval()

    correct = 0
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(), retrieve_steps=True)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        loss = criterion(out, data.y)  # Compute the loss.
        losses.append(loss.to('cpu').detach().numpy())
    return (correct / len(loader.dataset), np.mean(losses), out)  # Derive ratio of correct predictions.


np.random.seed(174)

# Setting parameters
max_epochs = 160
max_patience_counter = 30

# Setting up store results
save_models_all = {}

classes = pd.unique(target)

# For each repetition
for r in datasets:
    # Setting up the models
    model = FDiGNN_TAG(hidden_channels=64, drop=0.15, n_node_feat=len(node_attrs), K=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(
        f'LID_TAG_Models/LID_model_Graphlets5_4TAG_64HC15D001LR0001WD_{r}_{sys.argv[-1]}'))

    save_models_all[r] = model

print(f'Graphlets {sys.argv[-1]} starting explanations.')

def entropy_integral(input_x):
        num = input_x**2*(np.log(input_x) - np.log(1-input_x)) + (2*np.log(1-input_x)-1)*input_x-np.log(1-input_x)
        den = 2*np.log(2)
        x = (-num/den)
        x[input_x == 1] = np.float32(0.7213475204444817) # Force 1 to be the maximum integral
        x[input_x == 0] = np.float32(0) # Force 0 to be 0 integral
        return x

effect = {}
entropy = {}
all_preds ={}
for key in datasets:
    effect[key] = {}
    entropy[key] = {}
    model = save_models_all[key]
    all_preds[key] = {}
    # Normal Preds
    out_normal = pd.DataFrame()
    for data in DataLoader(all_data_full[key], batch_size=32, shuffle=False):  # Iterate in batches over the training/test dataset.
        model.eval()
        out = model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(), retrieve_steps=True)
        out = F.softmax(out, 1)
        out_normal = pd.concat((out_normal, pd.DataFrame(out.detach().cpu().numpy())))
    normal_entropy = entropy_integral(out_normal)
    all_preds[key]['Normal'] = out_normal.reset_index().iloc[:,1:].to_dict()
    for i in range(len(FDiN.nodes())):
        node = list(FDiN.nodes())[i]
        all_preds[key][node] = {}
        original_values = datasets[key].loc[:, node].copy().values
        original_feature_values = datasets_bin[key].loc[:, node].copy().values
        #shuffled_values = datasets[key].loc[:, node].copy().values
        #np.random.shuffle(shuffled_values)
        q_values = [0.05, 0.5, 0.95]
        effect[key][i] = pd.DataFrame(columns=q_values)
        entropy[key][i] = pd.DataFrame(columns=q_values)
        quantile_values = np.quantile(original_values, q=q_values)
        quantile_feature_values = np.quantile(original_feature_values, q=q_values)
        for q in range(len(quantile_values)):

            for g in range(len(all_data_full[key])):
                all_data_full[key][g].x[i, -2] = quantile_values[q]#shuffled_values[g]
                if quantile_feature_values[q] != 0:
                    if quantile_feature_values[q] != 1:
                        all_data_full[key][g].x[i, -1] = 1
                    else:
                        all_data_full[key][g].x[i, -1] = quantile_feature_values[q]
                else:
                    all_data_full[key][g].x[i, -1] = quantile_feature_values[q]

            out_shuffled = pd.DataFrame()
            for data in DataLoader(all_data_full[key], batch_size=32, shuffle=False):  # Iterate in batches over the training/test dataset.
                model.eval()
                out = model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(), retrieve_steps=True)
                out = F.softmax(out, 1)
                out_shuffled = pd.concat((out_shuffled, pd.DataFrame(out.detach().cpu().numpy())))
            all_preds[key][node][q] = out_shuffled.reset_index().iloc[:,1:].to_dict()

            results = pd.DataFrame((out_normal.values - out_shuffled.values)).abs()
            effect[key][i][q_values[q]] = results[0]

            shuffled_entropy = entropy_integral(out_shuffled)
            entropy[key][i][q_values[q]]= pd.DataFrame((shuffled_entropy.values - normal_entropy.values)).abs().sum(axis=1)

        # Restore values
        for g in range(len(all_data_full[key])):
            all_data_full[key][g].x[i, -2] = original_values[g]
            all_data_full[key][g].x[i, -1] = original_feature_values[g]
    print(key)


trimmed_pred_change = {}
pred_change = {}

new_df = pd.DataFrame(columns=range(len(sample_cols)))
for node in effect['Normal + Path'].keys():
    new_df.loc[node] = effect['Normal + Path'][node].max(axis=1).values#.sort_values().mean()
new_df.index = list(FDiN.nodes())
new_df = (new_df/new_df.sum()).replace({np.nan:0})
a = new_df.T.apply(
    lambda x: x.sort_values(ascending=False).values).T
pred_change['Normal + Path'] = a.median(axis=1).sort_values()
a = a.iloc[:,:int(1/3*len(new_df.columns))].median(axis=1).sort_values()#.head(20)
trimmed_pred_change['Normal + Path'] = a

new_df = pd.DataFrame(columns=range(len(sample_cols)))
for node in effect['Aleatorized + Path'].keys():
    new_df.loc[node] = effect['Aleatorized + Path'][node].max(axis=1).values#.sort_values().mean()
new_df.index = list(FDiN.nodes())
new_df = (new_df/new_df.sum()).replace({np.nan:0})
a = new_df.T.apply(
    lambda x: x.sort_values(ascending=False).values).T
pred_change['Aleatorized + Path'] = a.median(axis=1).sort_values()
a = a.iloc[:,:int(1/3*len(new_df.columns))].median(axis=1).sort_values()#.head(20)
trimmed_pred_change['Aleatorized + Path'] = a

trimmed_entropy_change = {}
entropy_change = {}

new_df = pd.DataFrame(columns=range(len(sample_cols)))
for node in entropy['Normal + Path'].keys():
    new_df.loc[node] = entropy['Normal + Path'][node].max(axis=1).values#.sort_values().mean()
new_df.index = list(FDiN.nodes())
new_df = (new_df/new_df.sum()).replace({np.nan:0})
a = new_df.T.apply(
    lambda x: x.sort_values(ascending=False).values).T
entropy_change['Normal + Path'] = a.median(axis=1).sort_values()
a = a.iloc[:,:int(1/3*len(new_df.columns))].median(axis=1).sort_values()#.head(20)
trimmed_entropy_change['Normal + Path'] = a

new_df = pd.DataFrame(columns=range(len(sample_cols)))
for node in entropy['Aleatorized + Path'].keys():
    new_df.loc[node] = entropy['Aleatorized + Path'][node].max(axis=1).values#.sort_values().mean()
new_df.index = list(FDiN.nodes())
new_df = (new_df/new_df.sum()).replace({np.nan:0})
a = new_df.T.apply(
    lambda x: x.sort_values(ascending=False).values).T
entropy_change['Aleatorized + Path'] = a.median(axis=1).sort_values()
a = a.iloc[:,:int(1/3*len(new_df.columns))].median(axis=1).sort_values()#.head(20)
trimmed_entropy_change['Aleatorized + Path'] = a


all_ranks = pd.concat((rank_imp_feats_rf.loc[pathway_to_alter + single_nodes, ['Normal + Path', 'Aleatorized + Path']], 
           rank_imp_feats_plsda.loc[pathway_to_alter + single_nodes, ['Normal + Path', 'Aleatorized + Path']], 
          pd.DataFrame(trimmed_pred_change).rank(ascending=False).loc[pathway_to_alter + single_nodes],
          pd.DataFrame(pred_change).rank(ascending=False).loc[pathway_to_alter + single_nodes],
          pd.DataFrame(trimmed_entropy_change).rank(ascending=False).loc[pathway_to_alter + single_nodes],
          pd.DataFrame(entropy_change).rank(ascending=False).loc[pathway_to_alter + single_nodes]), axis=1)

all_ranks.columns = ['RF - Normal', 'RF - Aleat.', 'PLSDA - Normal', 'PLSDA - Aleat.', 
                     'GNN - Normal - Trim.Pred', 'GNN - Aleat. - Trim.Pred', 'GNN - Normal - Pred',
                     'GNN - Aleat. - Pred', 'GNN - Normal - Trim.Entr.', 'GNN - Aleat. - Trim.Entr.',
                    'GNN - Normal - Entr.', 'GNN - Aleat. - Entr.']

all_ranks.to_excel(f'LID_TAG_Results/LID_Graphlets5_Normal_TAG_{sys.argv[-1]}.xlsx')

with open(f'LID_TAG_Results/LID_Graphlets5_All_Predictions_TAG_{sys.argv[-1]}.json', 'w') as f:
    json.dump(all_preds, f)

end_time = time.time()

print(f'Finished running Graphlets {sys.argv[-1]}. Time elapsed: {end_time - start_time}.')