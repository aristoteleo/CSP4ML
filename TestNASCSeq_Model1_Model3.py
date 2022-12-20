#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import warnings

warnings.filterwarnings('ignore')
import dynamo as dyn
import matplotlib

matplotlib.use('TkAgg')

filename = './data/NASC-seq/nasc-seq.h5ad'
nasc_seq = dyn.read(filename)

# select_cells = [x for x in nasc_seq.obs.Condition_labelled]
# nasc_seq = nasc_seq[select_cells, :]

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

dyn.tl.recipe_kin_data(adata=nasc_seq,
                       keep_filtered_genes=True,
                       keep_raw_layers=True,
                       del_2nd_moments=False,
                       tkey='time',
                       n_top_genes=1000,
                       )

dyn.tl.reduceDimension(nasc_seq, reduction_method='umap')
dyn.tl.cell_velocities(nasc_seq, enforce=True, vkey='velocity_T', ekey='M_t', basis='umap')
dyn.pl.streamline_plot(nasc_seq, color='Condition', basis='umap')
