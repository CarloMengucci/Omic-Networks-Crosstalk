
"""
Created on Sat Jul 30 10:10:04 2022

@author: carlo_mengucci
"""

"""
Example to build spearman correlation networks from merged dataset returned by process_NMR_and_omic_merge.py.
Graphs are created as networkx package objects and are saved in .gml format to be compatible with gephy and CytoScape.

Graphs for each sex are built for t0, grpahs for each sex and diet/intervention are built at te.

K-clique communities are defined and found according to Gergely Palla et al., doi:10.1038/nature03607

"""

import numpy as np

import pandas as pd
from os.path import join as pj
from pathlib import Path

import networkx as nx
import json
from networkx.algorithms import community


## Path Variables ##

data_dir=Path('Path/to/Data').resolve()
results_dir=Path('Path/to/Results').resolve()

merged_df_t0 = pd.read_csv(pj(data_dir, 'merged_final_t0.csv'), header=0,
                        sep=',', index_col=0) # Load csv files created by process_NMR_and_omicmerge.py

merged_df_te = pd.read_csv(pj(data_dir, 'merged_final_te.csv'), header=0,
                        sep=',', index_col=0)

meta_data = ['some','list','of','metadata', 'columns', 'in the dataset'] #create list of metadata  columns in the dataset

m_feats = [column for column in merged_df_t0.columns if column not in meta_data] #create list of proper feature columns in dataset

# create a list of labels to represent the type of features. This will be passed as node attribute in network construction

microb=np.repeat('microbiome', n_microbic_taxa)  # n_*type_of_feature: number of feature of that type in the merged dataset,
vol=np.repeat('volatilome', n_volatile_compounds) # ater processing
metab=np.repeat('metabolome', n_spectral_aggl)

types=np.concatenate((microb,vol,metab), axis=None)

# %% ## Computing adjacency matrix from correlations, create and save networks for M/F T0 ##

Graphs=[]

for sex in np.unique(merged_df_t0['sex'].values): # must have a column to identify sex in df

        group=sex
        types_dict=dict(zip(m_feats, types)) # dictionary of node attributes (feature types)
        corr=merged_df_t0[(merged_df_t0['sex'] == sex)].loc[:,m_feats].corr(method='spearman') #compute correlation map directly from
                                                                                               #dataframe, using pandas df builtin methods            

        adj=np.where(np.abs(corr) >= threshold, 1,0) ## transform adjacency matrix to binary (thresholding)
        np.fill_diagonal(adj,0)                      # the resulting network will be unweighted and undirected (not mandatory)

        G=nx.from_numpy_matrix(adj) #Create graph object from adjacency matrix using networkx
        names = dict(zip(list(G.nodes), m_feats)) #create dictionary for node names. Each node is a feature: get the names from columns
        G=nx.relabel_nodes(G, names)              #in the dataset and rename the nodes in the nx Graph object
        nx.set_node_attributes(G, types_dict, 'feature') #set feature type as attribute for the nodes: the result is a network with
                                                         #nodes named after each feature, with the the type of feature as attribute
        
        Graphs.append((G,sex)) #append a tuple to the Graphs empty array, containing the nx object (G) and a label describing sex
        
        nx.write_gml(G, pj(results_dir, 't0_net_%s.gml'%group)) #save each neatwork in .gml format, compatible with rendering tools such as gephi
                                                                #and CytoScape. Each file is automatically labeled.
#%% ## Computing adjacency matrix from correlations, create and save networks for M/F + treatment groups at Te ##

Graphs_te=[]

for sex in np.unique(merged_df_te['sex'].values): # must have a column to identify sex in df
    for diet in np.unique(merged_df_te['diet'].values): # must have a column to identify treatment/diet in df

        group=sex+diet
        types_dict=dict(zip(m_feats, types))
        corr=merged_df_te[(merged_df_te['sex'] == sex) & (merged_df_te['diet'] == diet)].loc[:,m_feats].corr(method='spearman')#compute correlation map directly from
                                                                                                                               #dataframe, using pandas df builtin methods   

        adj=np.where(np.abs(corr) >= threshold, 1,0) ## transform adjacency matrix to binary (thresholding)
        np.fill_diagonal(adj,0)                      # the resulting network will be unweighted and undirected (not mandatory)

        G=nx.from_numpy_matrix(adj)
        names = dict(zip(list(G.nodes), m_feats))
        G=nx.relabel_nodes(G, names)
        nx.set_node_attributes(G, types_dict, 'feature')
        Graphs_te.append((G,sex,diet)) ## append a tuple with graph, sex label, diet label

        nx.write_gml(G, 'te_net%s.gml'%group) #save each neatwork in .gml format, compatible with rendering tools such as gephi
                                              #and CytoScape. Each file is automatically labeled.

# %% ##find k-cliques communities and dump as json, example with te graphs ###

for graph in Graphs_te:
    G = graph[0] #first element in tuple Graphs_te= nx graph object
    label = graph[1]+graph[2] #second and third elements in tuple Graphs_te are sex and intervention for labelling
    
    k=list(community.k_clique_communities(G, k=order_of_the_clique)) #find all the communities of cliques of order k in G

    k_q=[sorted(list(k[i])) for i in range (len(k))] ## important: list comprehension for typecasting from FrozenSet
    
    f = open(pj(results_dir,'kq_communities_%s.txt'%label), 'w') #dump and save k_q communities for each network (sex,diet/intervention)
    json.dump(k_q, f)
    f.close()
