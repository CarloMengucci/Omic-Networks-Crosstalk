# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:26:47 2022

@author: carlo_mengucci
"""

"""
Example code using python packages to process 1HNMR spectral features build a merged dataframe with
microbic abundances and volotatile compounds from GC-MS

We assume that at this stage, all omic data have been preprocessed (normalized, scaled...) and microbic taxa and
volatile compounds have already been merged in a single dataset (merged_t0, merged_te in the example).
"""


import numpy as np

import pandas as pd
from os.path import join as pj
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
import collections
import json

## Path Variables ##

data_dir=Path('Path/to/Data').resolve() # ensure correct path resolution for different filesystems and architectures
results_dir=Path('Path/to/Results').resolve()

# %% ## data loading ##


## t0 
metab_t0 = pd.read_csv(pj(data_dir, 't0_metabolome.csv'), index_col = 'idx') # T0 datasets loading. It is advisable to load
merged_t0 = pd.read_csv(pj(data_dir, 't0_vol_microb_csv'), index_col = 'idx')# samples IDs as dataframe index

## te
metab_te = pd.read_csv(pj(data_dir, 'te_metabolome.csv'), index_col = 'idx')# T0 datasets loading. It is advisable to load
merged_te = pd.read_csv(pj(data_dir, 'te_vol_microb.csv'), index_col = 'idx')# samples IDs as dataframe index

# %% ## Inducing sparsity in spectral features, by targeting intervention classes at Te, with a l1 penalized logistic regression ##

LR = LogisticRegression(penalty = 'l1', dual = False, C = c_optimized, class_weight = 'balanced', solver = 'liblinear',
                        max_iter = 1000) #Initialize the model, hyperparameter C needs to be tuned

meta_data=['some','list','of','metadata', 'columns', 'in the dataset'] #create a list of metadata columns in the dataframes

m_feats=[column for column in metab_te.columns if column not in meta_data] # filter out metadata, keep only spectral buckets

LR.fit(X=StandardScaler().fit_transform(metab_te.loc[:, m_feats].values), y=metab_te['DIET']) #Fit LR model to Te data

coefs = LR.coef_ ## shape (coef) n_class X n_features, get matrices of coefficients of L1 penalized LR

filt = np.unique(np.nonzero(coefs)[1]) # get nonzero elements in column, select important variable for at least 1 class

m_imp_feats = [m_feats[i] for i in filt] # filter out important features from te, using non-zero elements of the LR coefficient matrix
                                        # and apply filter to t0 features

# %% ## Feature agglomeration: summarize spectral buckets into highly correlated clusters, using avg to represent the pattern (pooling f)
     ## of spectral features. Preserve info on agglomerates to comment intramolecular or long distance correlations##

FA = FeatureAgglomeration(distance_threshold = chosen_threshold, affinity = 'correlation', linkage = 'complete', n_clusters = None,
                          pooling_func = np.median) # Initialize feature agglomeration model. Distance threshold: maximum cophenetic
                                                    # distance above which clusters are not merged

## t0
agl_t0_attr = FA.fit(metab_t0.loc[:,m_imp_feats]) # Fit FA to save class attributes
agl_t0 = FA.fit_transform(metab_t0.loc[:,m_imp_feats]) # Fit FA and trnasform Spectral Data

## Create, sort and dump dictionary of feature agglomeration: preserve information for spectral matching ##

feats_agglomerated_t0 = dict(zip(m_imp_feats, agl_t0_attr.labels_))
feats_agglomerated_t0 = {str(n):[k for k in feats_agglomerated_t0.keys() if feats_agglomerated_t0[k] == n] for n in 
                          set(feats_agglomerated_t0.values())}

feats_agglomerated_t0 = collections.OrderedDict(sorted(feats_agglomerated_t0.items()))

f = open(pj(results_dir,'feats_agglomerated_t0.txt'), 'w')
json.dump(feats_agglomerated_t0, f)
f.close()

## Append agglomerated spectral features to original metabolome t0 dataset, then label these new columns ##

for i in range(len(agl_t0[0])):
    metab_t0['Spectral_AGG_%s'%i] = agl_t0[:,i]

agg_feats_label_t0 = [column for column in metab_t0.columns if '_AGG_' in column]

## te (same procedures)

agl_te_attr = FA.fit(metab_te.loc[:,m_imp_feats])
agl_te = FA.fit_transform(metab_te.loc[:,m_imp_feats])

feats_agglomerated_te = dict(zip(m_imp_feats, agl_te_attr.labels_))
feats_agglomerated_te = {str(n):[k for k in feats_agglomerated_te.keys() if feats_agglomerated_te[k] == n] for n in 
                          set(feats_agglomerated_te.values())}

## sort dict and dump feature agglomeration ##
feats_agglomerated_te = collections.OrderedDict(sorted(feats_agglomerated_te.items()))

f = open(pj(results_dir,'feats_agglomerated_te.txt'), 'w')
json.dump(feats_agglomerated_te, f)
f.close()


for i in range(len(agl_te[0])):
    metab_te['Spectral_AGG_%s'%i] = agl_te[:,i]
    
agg_feats_label_te = [column for column in metab_te.columns if '_AGG_' in column]

# %% %% ## Select class-related features in original datasets and merge##

metab_t0 = metab_t0.loc[:,meta_data+agg_feats_label_t0]
metab_te = metab_te.loc[:, meta_data+agg_feats_label_te]

merged_final_t0=merged_t0.merge(metab_t0, on= meta_data, how='inner') # inner merge on all metadata columns to enhance consistency
merged_final_te=merged_te.merge(metab_te, on=meta_data, how='inner')

## exporting to csv ##

merged_final_t0.to_csv(pj(data_dir, 'merged_final_t0.csv'), header = True, sep = ',')
merged_final_te.to_csv(pj(data_dir, 'merged_final_te.csv'), header = True, sep = ',')
