#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from storm_param_infer import storm_sto_knn

def dynamo_pp_workflow(adata, tkey='time', experiment_type='kin', n_top_genes=1000):
    from dynamo.preprocessing import Preprocessor
    from dynamo.preprocessing.utils import (
        del_raw_layers,
        detect_experiment_datatype,
        reset_adata_X,
    )
    from dynamo.tools.connectivity import neighbors, normalize_knn_graph
    from dynamo.preprocessing.pca import pca
    from dynamo.tl import moments

    keep_filtered_cells = False
    keep_filtered_genes = True
    keep_raw_layers = True
    del_2nd_moments = True
    reset_X = True

    has_splicing, has_labeling, splicing_labeling, _ = detect_experiment_datatype(adata)

    if has_splicing and has_labeling and splicing_labeling:
        layers = ["X_new", "X_total", "X_uu", "X_ul", "X_su", "X_sl"]
    elif has_labeling:
        layers = ["X_new", "X_total"]

    if not has_labeling:
        raise Exception(
            "This recipe is only applicable to kinetics experiment datasets that have "
            "labeling data (at least either with `'uu', 'ul', 'su', 'sl'` or `'new', 'total'` "
            "layers."
        )

    # Preprocessing
    preprocessor = Preprocessor(cell_cycle_score_enable=True)
    preprocessor.config_monocle_recipe(adata, n_top_genes=n_top_genes)
    preprocessor.size_factor_kwargs.update(
        {
            "X_total_layers": False,
            "splicing_total_layers": False,
        }
    )
    preprocessor.normalize_by_cells_function_kwargs.update(
        {
            "X_total_layers": False,
            "splicing_total_layers": False,
            "keep_filtered": keep_filtered_genes,
            "total_szfactor": "total_Size_Factor",
        }
    )
    preprocessor.filter_cells_by_outliers_kwargs["keep_filtered"] = keep_filtered_cells
    preprocessor.select_genes_kwargs["keep_filtered"] = keep_filtered_genes

    if reset_X:
        reset_adata_X(adata, experiment_type=experiment_type, has_labeling=has_labeling, has_splicing=has_splicing)
    preprocessor.preprocess_adata_monocle(adata=adata, tkey=tkey, experiment_type=experiment_type)

    if not keep_raw_layers:
        del_raw_layers(adata)

    if has_splicing and has_labeling:
        # new, total (and uu, ul, su, sl if existed) layers will be normalized with size factor calculated with total
        # layers spliced / unspliced layers will be normalized independently.

        tkey = adata.uns["pp"]["tkey"]
        # first calculate moments for labeling data relevant layers using total based connectivity graph
        moments(adata, group=tkey, layers=layers)
        storm_sto_knn(adata, group=tkey, layers=["new", "total", "uu", "ul", "su", "sl"], conn=adata.obsp["moments_con"])

        # then we want to calculate moments for spliced and unspliced layers based on connectivity graph from spliced
        # data.
        # first get X_spliced based pca embedding
        CM = np.log1p(adata[:, adata.var.use_for_pca].layers["X_spliced"].toarray())
        cm_genesums = CM.sum(axis=0)
        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
        valid_ind = np.array(valid_ind).flatten()

        pca(adata, CM[:, valid_ind], pca_key="X_spliced_pca")
        # then get neighbors graph based on X_spliced_pca
        neighbors(adata, X_data=adata.obsm["X_spliced_pca"], layer="X_spliced")
        # then normalize neighbors graph so that each row sums up to be 1
        conn = normalize_knn_graph(adata.obsp["connectivities"] > 0)
        # then calculate moments for spliced related layers using spliced based connectivity graph
        moments(adata, conn=conn, layers=["X_spliced", "X_unspliced"])
        storm_sto_knn(adata, conn=conn, layers=["spliced", "unspliced"])
    else:
        tkey = adata.uns["pp"]["tkey"]
        moments(adata, group=tkey, layers=layers)
        storm_sto_knn(adata, group=tkey, layers=['new', 'total'], conn=adata.obsp["moments_con"])