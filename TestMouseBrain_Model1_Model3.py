#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import scanpy as sc
import dynamo as dyn
import matplotlib

matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

filename = '/home/pqw/pythonProject3/data/qi_P7_raw_2.h5ad'

adata = sc.read(filename)

adata.obs['treatment'] = adata.obs['treatment'].map({"4sU-4hr": 4, "4sU-2hr": 2})

adata.obs['treatment'] = adata.obs['treatment'].astype(float)

adata.obs['batch'].value_counts()

adata.X.sum(1).shape

sc_adata = adata.copy()

sc.pp.recipe_seurat(sc_adata)


# dyn.pl.basic_stats(adata)
#
# dyn.pl.show_fraction(adata)


adata_oneshot = adata.copy()

dyn.tl.recipe_kin_data(adata_oneshot, tkey='treatment', n_top_genes=1000)

dyn.pl.streamline_plot(
    adata_oneshot,
    # color=["res.name", 'ntr'],
    color=["res.name"],
    ncols=4,
)
