#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csc_matrix
import scipy

# filename
exp2_new_filename = './data/NASC-seq/GSE128273_exp2_newcounts.csv'
exp2_total_filename = './data/NASC-seq/GSE128273_exp2_readcounts.csv'
exp3_new_filename = './data/NASC-seq/GSE128273_exp3_newcounts.csv'
exp3_total_filename = './data/NASC-seq/GSE128273_exp3_readcounts.csv'
exp4_new_filename = './data/NASC-seq/GSE128273_exp4_newcounts.csv'
exp4_total_filename = './data/NASC-seq/GSE128273_exp4_readcounts.csv'

# gene id
Gene_Id = pd.read_csv('./data/NASC-seq/GSE128273_exp2_newcounts.csv', header=0, index_col=0).index
var = pd.DataFrame(index=Gene_Id)
var["Gene_Id"] = Gene_Id
n_vars = len(var)

new_counts = csc_matrix(np.empty(shape=[0, n_vars]))
total_counts = csc_matrix(np.empty(shape=[0, n_vars]))
Cell_Id = []
Condition_labelled = []
Condition_stimulated = []
time = []

# total counts; time; cell_id; condition
for exp in ['2', '3', '4']:
    cur_exp = pd.read_csv(locals()['exp' + exp + '_total_filename'], header=0, index_col=0)

    cur_Cell_Id = cur_exp.columns.tolist()
    Cell_Id = Cell_Id + cur_Cell_Id

    if exp == '2':
        cur_time = len(cur_Cell_Id) * ['15']
    elif exp == '3':
        cur_time = len(cur_Cell_Id) * ['30']
    else:
        cur_time = len(cur_Cell_Id) * ['60']
    time = time + cur_time

    labelled_indicator = [True] * len(cur_Cell_Id)
    for i in range(len(cur_Cell_Id)):
        if 'unlabelled' in cur_Cell_Id[i]:
            labelled_indicator[i] = False
    Condition_labelled = Condition_labelled + labelled_indicator

    stimulated_indicator = [True] * len(cur_Cell_Id)
    for i in range(len(cur_Cell_Id)):
        if 'Unstimulated' in cur_Cell_Id[i]:
            stimulated_indicator[i] = False
    Condition_stimulated = Condition_stimulated + stimulated_indicator

    cur_exp = cur_exp.fillna(value=0)
    cur_counts = csc_matrix(cur_exp.T.to_numpy().astype(int))
    total_counts = scipy.sparse.vstack((total_counts, cur_counts))

# new counts
for exp in ['2', '3', '4']:
    cur_exp = pd.read_csv(locals()['exp' + exp + '_new_filename'], header=0, index_col=0)
    cur_exp = cur_exp.fillna(value=0)
    cur_counts = csc_matrix(cur_exp.T.to_numpy().astype(int))
    new_counts = scipy.sparse.vstack((new_counts, cur_counts))

obs = pd.DataFrame(index=Cell_Id)
obs["time"] = time
obs["Cell_Id"] = Cell_Id
obs["Condition_labelled"] = Condition_labelled
obs["Condition_stimulated"] = Condition_stimulated
obs["exp_type"] = ["Pulse"] * len(Cell_Id)

X = total_counts.tocsc()
adata = ad.AnnData(X, obs=obs, var=var)

adata.layers["new"] = new_counts.tocsc()
adata.layers["total"] = total_counts.tocsc()

adata.filename = './data/NASC-seq/nasc-seq.h5ad'

nasc_seq = adata
# select genes with monotonically increasing new/total ratio
nasc_seq_15 = nasc_seq[nasc_seq.obs.time == '15', :]
nasc_seq_30 = nasc_seq[nasc_seq.obs.time == '30', :]
nasc_seq_60 = nasc_seq[nasc_seq.obs.time == '60', :]
nasc_seq_15_new = np.mean(nasc_seq_15.layers['new'], axis=0)
nasc_seq_15_total = np.mean(nasc_seq_15.layers['total'], axis=0)
nasc_seq_30_new = np.mean(nasc_seq_30.layers['new'], axis=0)
nasc_seq_30_total = np.mean(nasc_seq_30.layers['total'], axis=0)
nasc_seq_60_new = np.mean(nasc_seq_60.layers['new'], axis=0)
nasc_seq_60_total = np.mean(nasc_seq_60.layers['total'], axis=0)

nasc_seq_15_new_total_ratio = np.array(nasc_seq_15_new / nasc_seq_15_total)
nasc_seq_30_new_total_ratio = np.array(nasc_seq_30_new / nasc_seq_30_total)
nasc_seq_60_new_total_ratio = np.array(nasc_seq_60_new / nasc_seq_60_total)

temp_15 = nasc_seq_15_new_total_ratio[0, :]
temp_30 = nasc_seq_30_new_total_ratio[0, :]
temp_60 = nasc_seq_60_new_total_ratio[0, :]

increase_gene_index = np.logical_and(temp_15 < temp_30, temp_30 < temp_60)
print(np.sum(increase_gene_index))
np.savetxt('./data/NASC-seq/nasc-seq_gene.csv', increase_gene_index, delimiter=',')
