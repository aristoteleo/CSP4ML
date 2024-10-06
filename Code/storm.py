#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from dynamo.estimation.tsc.twostep import (
    lin_reg_gamma_synthesis,
    fit_slope_stochastic
)

from storm_param_infer import (
    MLE_Cell_Specific_Poisson_SS,
    MLE_Cell_Specific_Poisson,
    MLE_Cell_Specific_Zero_Inflated_Poisson,
    MLE_Independent_Cell_Specific_Poisson,
    Cell_Specific_Alpha_Beta,
    MLE_ICSP_Without_SS,
    Select_SCV_Genes
)

def storm_kin_data(adata, use_genes=None, tkey='time', assumption='steady_state', method='CSP_Baseline'):

    subset_adata = adata[:, use_genes].copy()
    gene_indices = [list(adata.var_names).index(item) if item in list(adata.var_names) else -1 for item in list(use_genes)]
    time = subset_adata.obs[tkey]

    # Initialization based on the steady-state assumption
    if method is not 'CSP_Splicing':
        layers_smoothed = ["M_t", "M_n"]
        Total_smoothed, New_smoothed = (
            subset_adata.layers[layers_smoothed[0]].T,
            subset_adata.layers[layers_smoothed[1]].T,
        )
        (gamma_init, _, _, _, _,) = lin_reg_gamma_synthesis(Total_smoothed, New_smoothed, time, perc_right=5)

        # Read raw counts
        layers_raw = ["total", "new"]
        Total_raw, New_raw = (
            subset_adata.layers[layers_raw[0]].T,
            subset_adata.layers[layers_raw[1]].T,
        )

        # Read smoothed values based CSP type distribution for cell-specific parameter inference
        layers_smoothed_CSP = ["M_CSP_t", "M_CSP_n"]
        Total_smoothed_CSP, New_smoothed_CSP = (
            subset_adata.layers[layers_smoothed_CSP[0]].T,
            subset_adata.layers[layers_smoothed_CSP[1]].T,
        )

        # Parameters inference based on maximum likelihood estimation
        cell_total = subset_adata.obs['initial_cell_size'].astype("float").values
    else:
        layers_smoothed = ["M_u", "M_s", "M_t", "M_n"]
        U_smoothed, S_smoothed, Total_smoothed, New_smoothed = (
            subset_adata.layers[layers_smoothed[0]].T,
            subset_adata.layers[layers_smoothed[1]].T,
            subset_adata.layers[layers_smoothed[2]].T,
            subset_adata.layers[layers_smoothed[3]].T,
        )

        US_smoothed, S2_smoothed = (
            subset_adata.layers["M_us"].T,
            subset_adata.layers["M_ss"].T,
        )
        (gamma_k, _, _, _,) = fit_slope_stochastic(S_smoothed, U_smoothed, US_smoothed, S2_smoothed, perc_left=None,
                                                   perc_right=5)
        (gamma_init, _, _, _, _) = lin_reg_gamma_synthesis(Total_smoothed, New_smoothed, time, perc_right=5)
        beta_init = gamma_init / gamma_k  # gamma_k = gamma / beta

        # Read raw counts
        layers_raw = ["ul", "sl"]
        UL_raw, SL_raw = (
            subset_adata.layers[layers_raw[0]].T,
            subset_adata.layers[layers_raw[1]].T,
        )

        # Read smoothed values based CSP type distribution for cell-specific parameter inference
        UL_smoothed_CSP, SL_smoothed_CSP = (
            subset_adata.layers['M_CSP_ul'].T,
            subset_adata.layers['M_CSP_sl'].T,
        )

        # Parameters inference based on maximum likelihood estimation
        cell_total = subset_adata.obs['initial_cell_size'].astype("float").values

    # Parameter inference and RNA velocity
    if assumption == 'steady_state':
        method = 'CSP_Baseline'
        gamma, select_genes, gamma_r2, alpha = MLE_Cell_Specific_Poisson_SS(Total_raw, New_raw, time, gamma_init,
                                                                            cell_total, Total_smoothed, New_smoothed)
        k = 1 - np.exp(-gamma[:, None] * time[None, :])
        cell_wise_alpha = csr_matrix(gamma[:, None]).multiply(New_smoothed_CSP).multiply(1 / k)  # cell-specific alpha
        velocity_T = cell_wise_alpha - csr_matrix(gamma[:, None]).multiply(Total_smoothed)

        adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma'][gene_indices] = gamma
        adata.var['gamma_r2'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma_r2'][gene_indices] = gamma_r2
        adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
        adata.var['alpha'][gene_indices] = alpha
        adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
        adata.var['select_genes'][gene_indices] = select_genes

        adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
        adata.layers["velocity_T"][:, gene_indices] = velocity_T

        adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(cell_wise_alpha, dtype=np.float64).T
        adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha
    else:
        if method == 'CSP_Baseline':
            gamma, select_genes, gamma_r2, alpha = MLE_Cell_Specific_Poisson(New_raw, time, gamma_init, cell_total,
                                                                             Total_smoothed)
            k = 1 - np.exp(-gamma[:, None] * time[None, :])
            cell_wise_alpha = csr_matrix(gamma[:, None]).multiply(New_smoothed_CSP).multiply(1 / k)  # cell-specific alpha
            velocity_T = cell_wise_alpha - csr_matrix(gamma[:, None]).multiply(Total_smoothed)

            adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma'][gene_indices] = gamma
            adata.var['gamma_r2'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma_r2'][gene_indices] = gamma_r2
            adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
            adata.var['alpha'][gene_indices] = alpha
            adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
            adata.var['select_genes'][gene_indices] = select_genes

            adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
            adata.layers["velocity_T"][:, gene_indices] = velocity_T

            adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(
                cell_wise_alpha, dtype=np.float64).T
            adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha
        elif method == 'CSP_Splicing':
            gamma_s, select_genes, beta, gamma_t, gamma_r2, alpha = MLE_Independent_Cell_Specific_Poisson(UL_raw,
                                                                                                          SL_raw, time,
                                                                                                          gamma_init,
                                                                                                          beta_init,
                                                                                                          cell_total,
                                                                                                          Total_smoothed,
                                                                                                          S_smoothed)
            # Cell specific parameters (fixed gamma_s)
            cell_wise_alpha, cell_wise_beta = Cell_Specific_Alpha_Beta(UL_smoothed_CSP, SL_smoothed_CSP, time, gamma_s,
                                                                       beta)
            velocity_T = cell_wise_alpha - csr_matrix(gamma_s[:, None]).multiply(S_smoothed)
            velocity_S = cell_wise_beta.multiply(U_smoothed) - csr_matrix(gamma_s[:, None]).multiply(S_smoothed)
            # # Cell specific parameters(fixed gamma_t)
            # k = 1 - np.exp(-gamma_t[:, None] * time[None, :])
            # cell_wise_alpha = csr_matrix(gamma_t[:, None]).multiply(UL_smoothed_CSP+SL_smoothed_CSP).multiply(1 / k)
            # velocity_T = cell_wise_alpha - csr_matrix(gamma_t[:, None]).multiply(Total_smoothed)

            adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma'][gene_indices] = gamma_t
            adata.var['gamma_s'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma_s'][gene_indices] = gamma_s
            adata.var['gamma_r2'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma_r2'][gene_indices] = gamma_r2
            adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
            adata.var['alpha'][gene_indices] = alpha
            adata.var['beta'] = np.zeros(adata.n_vars) * np.nan
            adata.var['beta'][gene_indices] = beta
            adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
            adata.var['select_genes'][gene_indices] = select_genes

            adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
            adata.layers["velocity_T"][:, gene_indices] = velocity_T

            adata.layers["velocity_S"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            velocity_S = velocity_S.T.tocsr() if sp.issparse(velocity_S) else sp.csr_matrix(velocity_S, dtype=np.float64).T
            adata.layers["velocity_S"][:, gene_indices] = velocity_S

            adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(
                cell_wise_alpha, dtype=np.float64).T
            adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha

            adata.layers["cell_wise_beta"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            cell_wise_beta = cell_wise_beta.T.tocsr() if sp.issparse(cell_wise_beta) else sp.csr_matrix(
                cell_wise_beta, dtype=np.float64).T
            adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_beta

        elif method == 'CSP_Switching':
            gamma, prob_off, select_genes, gamma_r2, alpha = MLE_Cell_Specific_Zero_Inflated_Poisson(New_raw, time,
                                                                                                     gamma_init,
                                                                                                     cell_total)
            alpha = alpha * (1 - prob_off)
            k = 1 - np.exp(-gamma[:, None] * time[None, :])
            cell_wise_alpha = csr_matrix(gamma[:, None]).multiply(New_smoothed_CSP).multiply(1 / k) # cell-specific alpha*(1-p_off)
            velocity_T = cell_wise_alpha - csr_matrix(gamma[:, None]).multiply(Total_smoothed)

            adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma'][gene_indices] = gamma
            adata.var['gamma_r2'] = np.zeros(adata.n_vars) * np.nan
            adata.var['gamma_r2'][gene_indices] = gamma_r2
            adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
            adata.var['alpha'][gene_indices] = alpha
            adata.var['prob_off'] = np.zeros(adata.n_vars) * np.nan
            adata.var['prob_off'][gene_indices] = prob_off
            adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
            adata.var['select_genes'][gene_indices] = select_genes

            adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
            adata.layers["velocity_T"][:, gene_indices] = velocity_T

            adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
            cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(cell_wise_alpha, dtype=np.float64).T
            adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha

def storm_one_shot_data(adata, use_genes=None, tkey='time', assumption='steady_state', method='CSP_Baseline'):
    subset_adata = adata[:, use_genes].copy()
    gene_indices = [list(adata.var_names).index(item) if item in list(adata.var_names) else -1 for item in
                    list(use_genes)]
    time = subset_adata.obs[tkey]

    # Initialization based on the steady-state assumption
    if method is not 'CSP_Splicing':
        layers_smoothed = ["M_t", "M_n"]
        Total_smoothed, New_smoothed = (
            subset_adata.layers[layers_smoothed[0]].T,
            subset_adata.layers[layers_smoothed[1]].T,
        )
        (gamma_init, _, _, _, _,) = lin_reg_gamma_synthesis(Total_smoothed, New_smoothed, time, perc_right=5)

        # Read raw counts
        layers_raw = ["total", "new"]
        Total_raw, New_raw = (
            subset_adata.layers[layers_raw[0]].T,
            subset_adata.layers[layers_raw[1]].T,
        )

        # Read smoothed values based CSP type distribution for cell-specific parameter inference
        layers_smoothed_CSP = ["M_CSP_t", "M_CSP_n"]
        Total_smoothed_CSP, New_smoothed_CSP = (
            subset_adata.layers[layers_smoothed_CSP[0]].T,
            subset_adata.layers[layers_smoothed_CSP[1]].T,
        )

        # Parameters inference based on maximum likelihood estimation
        cell_total = subset_adata.obs['initial_cell_size'].astype("float").values
    else:
        # import scvelo as scv
        # subset_adata.layers['Ms'] = subset_adata.layers['M_s']
        # subset_adata.layers['Mu'] = subset_adata.layers['M_u']
        # scv.tl.recover_dynamics(subset_adata, var_names='all')
        scv_gamma = subset_adata.var.fit_gamma.values
        scv_beta = subset_adata.var.fit_beta.values
        scv_alpha = subset_adata.var.fit_alpha.values
        scv_t_switch = subset_adata.var.fit_t_.values
        scv_time = subset_adata.layers['fit_t'].T

        layers_smoothed = ["M_u", "M_s", "M_t", "M_n"]
        U_smoothed, S_smoothed, Total_smoothed, New_smoothed = (
            subset_adata.layers[layers_smoothed[0]].T,
            subset_adata.layers[layers_smoothed[1]].T,
            subset_adata.layers[layers_smoothed[2]].T,
            subset_adata.layers[layers_smoothed[3]].T,
        )
        (gamma_init, _, _, _, _) = lin_reg_gamma_synthesis(Total_smoothed, New_smoothed, time, perc_right=5)

        # Read raw counts
        layers_raw = ["ul", "sl"]
        UL_raw, SL_raw = (
            subset_adata.layers[layers_raw[0]].T,
            subset_adata.layers[layers_raw[1]].T,
        )

        # Read smoothed values based CSP type distribution for cell-specific parameter inference
        UL_smoothed_CSP, SL_smoothed_CSP = (
            subset_adata.layers['M_CSP_ul'].T,
            subset_adata.layers['M_CSP_sl'].T,
        )

        cell_total = subset_adata.obs['initial_cell_size'].astype("float").values


    # Parameter inference and RNA velocity
    if assumption == 'steady_state':
        method = 'CSP_Baseline'
        gamma, select_genes, gamma_r2, alpha = MLE_Cell_Specific_Poisson_SS(Total_raw, New_raw, time, gamma_init,
                                                                            cell_total, Total_smoothed, New_smoothed)
        k = 1 - np.exp(-gamma[:, None] * time[None, :])
        cell_wise_alpha = csr_matrix(gamma[:, None]).multiply(New_smoothed_CSP).multiply(1 / k)  # cell-specific alpha
        velocity_T = cell_wise_alpha - csr_matrix(gamma[:, None]).multiply(Total_smoothed)

        adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma'][gene_indices] = gamma
        adata.var['gamma_r2'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma_r2'][gene_indices] = gamma_r2
        adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
        adata.var['alpha'][gene_indices] = alpha
        adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
        adata.var['select_genes'][gene_indices] = select_genes

        adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
        adata.layers["velocity_T"][:, gene_indices] = velocity_T

        adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(cell_wise_alpha, dtype=np.float64).T
        adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha
    else:
        method = 'CSP_Splicing'
        alpha, beta, gamma_s, select_genes, gamma_t = MLE_ICSP_Without_SS(UL_raw, SL_raw, time, cell_total, scv_gamma,
                                                                          scv_beta, U_smoothed, S_smoothed, gamma_init,
                                                                          scv_t_switch, scv_time)
        Select_SCV_Genes(subset_adata)

        # # Cell specific parameters (fixed gamma_s)
        # alpha, beta = CSP4ML.cell_specific_alpha_beta(UL_smoothed_CSP, SL_smoothed_CSP, time,
        #                                               gamma_s, beta)
        # Cell specific parameters(fixed gamma_t)
        k = 1 - np.exp(-gamma_t[:, None] * time[None, :])
        cell_wise_alpha = csr_matrix(gamma_t[:, None]).multiply(UL_smoothed_CSP + SL_smoothed_CSP).multiply(1 / k)
        velocity_T = cell_wise_alpha - csr_matrix(gamma_s[:, None]).multiply(S_smoothed)
        velocity_S = csr_matrix(beta[:, None]).multiply(U_smoothed) - csr_matrix(gamma_s[:, None]).multiply(S_smoothed)

        adata.var['gamma'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma'][gene_indices] = gamma_t
        adata.var['gamma_s'] = np.zeros(adata.n_vars) * np.nan
        adata.var['gamma_s'][gene_indices] = gamma_s
        adata.var['select_genes'] = np.zeros(adata.n_vars, dtype=bool)
        adata.var['select_genes'][gene_indices] = select_genes
        adata.var['alpha'] = np.zeros(adata.n_vars) * np.nan
        adata.var['alpha'][gene_indices] = alpha
        adata.var['beta'] = np.zeros(adata.n_vars) * np.nan
        adata.var['beta'][gene_indices] = beta

        adata.var['no_linear_r2'] = np.zeros(adata.n_vars) * np.nan
        adata.var['no_linear_r2'][gene_indices] = subset_adata.var['no_linear_r2'].copy()


        adata.layers["velocity_T"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        velocity_T = velocity_T.T.tocsr() if sp.issparse(velocity_T) else sp.csr_matrix(velocity_T, dtype=np.float64).T
        adata.layers["velocity_T"][:, gene_indices] = velocity_T

        adata.layers["velocity_S"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        velocity_S = velocity_S.T.tocsr() if sp.issparse(velocity_S) else sp.csr_matrix(velocity_S, dtype=np.float64).T
        adata.layers["velocity_S"][:, gene_indices] = velocity_S

        adata.layers["cell_wise_alpha"] = sp.csr_matrix((adata.shape), dtype=np.float64)
        cell_wise_alpha = cell_wise_alpha.T.tocsr() if sp.issparse(cell_wise_alpha) else sp.csr_matrix(
            cell_wise_alpha, dtype=np.float64).T
        adata.layers["cell_wise_alpha"][:, gene_indices] = cell_wise_alpha