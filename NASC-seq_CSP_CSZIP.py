#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
import dynamo as dyn
import numpy as np

filename = './data/NASC-seq/nasc-seq.h5ad'
nasc_seq = dyn.read(filename)

color_map = []
i = 0
for (x, y) in zip(nasc_seq.obs.Condition_labelled, nasc_seq.obs.Condition_stimulated):
    if x and y:
        color_map = color_map + ['L;S;' + nasc_seq.obs.time[i]]
    elif x and (not y):
        color_map = color_map + ['L;US;' + nasc_seq.obs.time[i]]
    else:
        color_map = color_map + ['UL;US;' + nasc_seq.obs.time[i]]
    i = i + 1
nasc_seq.obs['Condition'] = color_map

nasc_seq.obs['time'] = nasc_seq.obs['time'].astype(str)
nasc_seq.obs['time'] = nasc_seq.obs['time'].astype(float)

nasc_seq.obs.time = nasc_seq.obs.time / 60  # convert minutes to hours

# reads genes with monotonically increasing new/total ratio
temp = np.loadtxt('./data/NASC-seq/nasc-seq_gene.csv', delimiter=',')
index = (temp > 0.01).tolist()
used_genes = nasc_seq.var[index]['Gene_Id'].values.tolist()

dyn.preprocessing.recipe_monocle(
    nasc_seq,
    tkey="time",
    experiment_type="kin",
    keep_filtered_genes=True,
    keep_raw_layers=True,
    genes_to_use=used_genes,
)

dyn.tl.dynamics(
    nasc_seq,
    model="deterministic",
    est_method="CSP4ML_CSP",
    # est_method="CSP4ML_CSZIP",
    del_2nd_moments=True,
)

dyn.tl.reduceDimension(nasc_seq, reduction_method='umap')
dyn.tl.cell_velocities(nasc_seq, enforce=True, vkey='velocity_T', ekey='M_t', basis='umap')
dyn.pl.streamline_plot(nasc_seq, color='Condition', basis='umap')


