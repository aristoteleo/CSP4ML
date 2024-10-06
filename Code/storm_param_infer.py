#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from typing import Tuple, Union, Optional, List

import scipy.optimize
from anndata import AnnData
import warnings

from scipy.sparse import (
    csr_matrix,
    issparse,
    SparseEfficiencyWarning,
)
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.special import gammaln
from scipy.optimize import root, fsolve

from dynamo.tools.utils import find_extreme

def MLE_Cell_Specific_Poisson_SS(
        R: Union[np.ndarray, csr_matrix],
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray,
        Total_smoothed,
        New_smoothed,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on the cell specific Poisson model using maximum likelihood estimation under the
    steady-state assumption

    Args:
        R: The number of total mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        select_genes: Genes selected according to R2. shape: (n_var,).
        gamma_r2: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).

    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    cell_capture_rate = cell_total / np.median(cell_total)

    # When there is only one labeling duration we can obtain the analytical solution directly but cannot define the
    # goodness-of-fit.
    if len(np.unique(time)) == 1:
        gamma = np.zeros(n_var)
        select_genes = np.ones(n_var, dtype=bool)
        gamma_r2 = np.ones(n_var)
        alpha = np.zeros(n_var)
        for i, r, n, r_smooth, n_smooth in tqdm(
                zip(np.arange(n_var), R, N, Total_smoothed, New_smoothed),
                "Infer parameters via maximum likelihood estimation based on the CSP model under the steady-state assumption"
        ):
            n = n.A.flatten() if issparse(n) else n.flatten()
            r = r.A.flatten() if issparse(r) else r.flatten()
            n_smooth = n_smooth.A.flatten() if issparse(n_smooth) else n_smooth.flatten()
            r_smooth = r_smooth.A.flatten() if issparse(r_smooth) else r_smooth.flatten()
            t_unique = np.unique(time)
            # mask = find_extreme(n_smooth, r_smooth, perc_left=None, perc_right=100)
            mask = find_extreme(n_smooth, r_smooth, perc_left=None, perc_right=50)
            gamma[i] = - np.log(1 - np.mean(n[mask]) / np.mean(r[mask])) / t_unique
            alpha[i] = gamma[i] * np.mean(r[mask]) / np.mean(cell_capture_rate[mask])
    else:
        gamma = np.zeros(n_var)
        select_genes = np.zeros(n_var, dtype=bool)
        gamma_r2 = np.zeros(n_var)
        alphadivgamma = np.zeros(n_var)
        for i, r, n, r_smooth, n_smooth in tqdm(
                zip(np.arange(n_var), R, N, Total_smoothed, New_smoothed),
                "Infer parameters via maximum likelihood estimation based on the CSP model under the steady-state assumption"
        ):
            n = n.A.flatten() if issparse(n) else n.flatten()
            r = r.A.flatten() if issparse(r) else r.flatten()

            n_smooth = n_smooth.A.flatten() if issparse(n_smooth) else n_smooth.flatten()
            r_smooth = r_smooth.A.flatten() if issparse(r_smooth) else r_smooth.flatten()
            mask = find_extreme(n_smooth, r_smooth, perc_left=None, perc_right=10)
            mask = [True] * n.shape[0]
            n = n[mask]
            r = r[mask]
            o = r-n

            def loss_func_ss(parameters):
                # Loss function of cell specific Poisson model under the steady-state assumption
                parameter_alpha_div_gamma, parameter_gamma = parameters
                mu_new = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time[mask])) * cell_capture_rate[mask]
                loss_new = -np.sum(n * np.log(mu_new) - mu_new)
                mu_old = parameter_alpha_div_gamma * cell_capture_rate[mask] * np.exp(-parameter_gamma * time[mask])
                loss_old = -np.sum(o * np.log(mu_old) - mu_old)
                loss = loss_new + loss_old
                return loss

            # Initialize and add boundary conditions
            alpha_div_gamma_init = np.mean(n) / np.mean(cell_capture_rate[mask] * (1 - np.exp(-gamma_init[i] * time[mask])))
            b1 = (0, 10 * alpha_div_gamma_init)
            b2 = (0, 10 * gamma_init[i])
            bnds = (b1, b2)
            parameters_init = np.array([alpha_div_gamma_init, gamma_init[i]])

            # Solve
            res = minimize(loss_func_ss, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2,
                           options={'maxiter': 1000})
            parameters = res.x
            loss = res.fun
            success = res.success
            alphadivgamma[i], gamma[i] = parameters
            if not success:
                print(res.message)

            # Calculate deviance R2 as goodness of fit
            def null_loss_func_ss(parameters_null):
                # Loss function of null model under the steady-state assumption
                parameters_a0_new, parameters_a0_old = parameters_null
                mu_new = parameters_a0_new * cell_capture_rate[mask]
                loss0_new = -np.sum(n * np.log(mu_new) - mu_new)
                mu_old = parameters_a0_old * cell_capture_rate[mask]
                loss0_old = -np.sum(o * np.log(mu_old) - mu_old)
                loss0 = loss0_new + loss0_old
                return loss0

            def saturated_loss_func_ss():
                # Loss function of saturated model under the steady-state assumption
                loss_saturated_new = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
                loss_saturated_old = -np.sum(o[o > 0] * np.log(o[o > 0]) - o[o > 0])
                loss_saturated = loss_saturated_new + loss_saturated_old
                return loss_saturated

            a0_new = np.mean(n) / np.mean(cell_capture_rate[mask])
            a0_old = np.mean(o) / np.mean(cell_capture_rate[mask])
            loss0 = null_loss_func_ss((a0_new, a0_old))

            loss_saturated = saturated_loss_func_ss()
            null_devanice = 2 * (loss0 - loss_saturated)
            devanice = 2 * (loss - loss_saturated)
            gamma_r2[i] = 1 - (devanice / (2 * n_obs - 2)) / (null_devanice / (2 * n_obs - 2))

        # Top 40% genes were selected by goodness of fit
        number_selected_genes = int(n_var * 0.4)
        gamma_r2[gamma < 0.01] = 0
        sort_index = np.argsort(-gamma_r2)
        select_genes[sort_index[:number_selected_genes]] = 1
        select_genes[sort_index[number_selected_genes + 1:]] = 0

        alpha = alphadivgamma * gamma

    return gamma, select_genes, gamma_r2, alpha


def MLE_Cell_Specific_Poisson(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray,
        R,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on cell specific Poisson distributions using maximum likelihood estimation

    Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        select_genes: Genes selected according to R2. shape: (n_var,).
        gamma_r2: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    select_genes= np.zeros(n_var,dtype=bool)
    gamma_r2 = np.zeros(n_var)
    alphadivgamma = np.zeros(n_var)
    for i, n, r in tqdm(
            zip(np.arange(n_var), N, R),
            "Infer parameters via maximum likelihood estimation based on the CSP model"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        r = r.A.flatten() if issparse(r) else r.flatten()

        def loss_func(parameters):
            # Loss function of cell specific Poisson model
            parameter_alpha_div_gamma, parameter_gamma = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            loss = -np.sum(n * np.log(mu) - mu)
            return loss

        # Initialize and add boundary conditions
        if ~np.isfinite(gamma_init[i]) or np.isnan(gamma_init[i]):
            gamma_init[i] = 0.5
            gamma[i], select_genes[i], gamma_r2[i], alphadivgamma[i] = 0.0, 0.0, 0.0, 0.0
            continue
        alpha_div_gamma_init = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
        b1 = (0, 10 * alpha_div_gamma_init)
        b2 = (0, 10 * gamma_init[i])
        bnds = (b1, b2)
        parameters_init = np.array([alpha_div_gamma_init, gamma_init[i]])

        # Solve
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alphadivgamma[i], gamma[i] = parameters
        if not success:
            print(res.message)

        # Calculate deviance R2 as goodness of fit
        def null_loss_func(parameters_null):
            # Loss function of null model
            parameters_a0 = parameters_null
            mu = parameters_a0 * cell_capture_rate
            loss0 = -np.sum(n * np.log(mu) - mu)
            return loss0

        def saturated_loss_func():
            # Loss function of saturated model
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        a0 = np.mean(n) / np.mean(cell_capture_rate)
        loss0 = null_loss_func(a0)

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))

    # Top 40% genes were selected by goodness of fit
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    select_genes[sort_index[:number_selected_genes]] = 1
    select_genes[sort_index[number_selected_genes + 1:]] = 0

    return gamma, select_genes, gamma_r2, alphadivgamma * gamma


def MLE_Cell_Specific_Zero_Inflated_Poisson(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        cell_total: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on cell specific zero-inflated Poisson distributions using maximum likelihood estimation

        Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).

    Returns:
        gamma: The estimated total mRNA degradation rate gamma. shape: (n_var,).
        prob_off: The estimated probability of gene expression being in the off state $p_{off}$. shape: (n_var,).
        select_genes: Genes selected according to R2. shape: (n_var,).
        gamma_r2_raw: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    select_genes = np.zeros(n_var, dtype=bool)
    gamma_r2 = np.zeros(n_var)
    prob_off = np.zeros(n_var)
    alphadivgamma = np.zeros(n_var)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Infer parameters via maximum likelihood estimation based on the CSZIP model"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of cell specific zero-inflated Poisson model
            parameter_alpha_div_gamma, parameter_gamma, parameter_prob_off = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            n_eq_0_index = n < 0.001
            n_over_0_index = n > 0.001
            loss_eq0 = -np.sum(np.log(parameter_prob_off + (1 - parameter_prob_off) * np.exp(-mu[n_eq_0_index])))
            loss_over0 = -np.sum(np.log(1 - parameter_prob_off) + (-mu[n_over_0_index]) + n[n_over_0_index] * np.log(
                mu[n_over_0_index]))
            loss = loss_eq0 + loss_over0
            return loss

        # Initialize and add boundary conditions
        mean_n = np.mean(n)
        s2_n = np.mean(np.power(n, 2))
        temp = np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
        prob_off_init = 1 - mean_n * mean_n * np.mean(
            np.power(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)), 2)) / (
                                temp * temp * (s2_n - mean_n))  # Use moment estimation as the initial value of prob_off
        alphadivgamma_init = mean_n / ((1 - prob_off_init) * temp)
        b1 = (0, 10 * alphadivgamma_init)
        b2 = (0, 10 * gamma_init[i])
        b3 = (0, (np.sum(n < 0.001) / np.sum(n > -1)))
        bnds = (b1, b2, b3)
        parameters_init = np.array([alphadivgamma_init, gamma_init[i], prob_off_init])

        # Slove
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        alphadivgamma[i], gamma[i], prob_off[i] = parameters
        loss = res.fun
        success = res.success
        if not success:
            print(res.message)

        # Calculate deviance R2 as goodness of fit

        def null_Loss_func(parameters_null):
            # Loss function of null model
            parameters_null_lambda, parameters_null_prob_off = parameters_null
            mu = parameters_null_lambda * cell_capture_rate
            n_eq_0_index = n < 0.0001
            n_over_0_index = n > 0.0001
            null_loss_eq0 = -np.sum(
                np.log(parameters_null_prob_off + (1 - parameters_null_prob_off) * np.exp(-mu[n_eq_0_index])))
            null_loss_over0 = -np.sum(
                np.log(1 - parameters_null_prob_off) + (-mu[n_over_0_index]) + n[n_over_0_index] * np.log(
                    mu[n_over_0_index]))
            null_loss = null_loss_eq0 + null_loss_over0
            return null_loss

        mean_cell_capture_rate = np.mean(cell_capture_rate)
        prob_off_init_null = 1 - mean_n * mean_n * np.mean(np.power(cell_capture_rate, 2)) / (
                mean_cell_capture_rate * mean_cell_capture_rate * (s2_n - mean_n))
        lambda_init_null = mean_n / ((1 - prob_off_init_null) * mean_cell_capture_rate)
        b1_null = (0, 10 * lambda_init_null)
        b2_null = (0, (np.sum(n < 0.001) / np.sum(n > -1)))
        bnds_null = (b1_null, b2_null)
        parameters_init_null = np.array([lambda_init_null, prob_off_init_null])
        res_null = minimize(null_Loss_func, parameters_init_null, method='SLSQP', bounds=bnds_null, tol=1e-2,
                            options={'maxiter': 1000})
        loss0 = res_null.fun

        def saturated_loss_func():
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)

        gamma_r2[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))

    # Top 40% genes were selected by goodness of fit
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    select_genes[sort_index[:number_selected_genes]] = 1
    select_genes[sort_index[number_selected_genes + 1:]] = 0

    return gamma, prob_off, select_genes, gamma_r2, gamma * alphadivgamma

def MLE_Independent_Cell_Specific_Poisson(
        UL: Union[np.ndarray, csr_matrix],
        SL: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        beta_init: np.ndarray,
        cell_total: np.ndarray,
        Total_smoothed: Union[np.ndarray, csr_matrix],
        S_smoothed: Union[np.ndarray, csr_matrix]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """"Infer parameters based on independent cell specific Poisson distributions using maximum likelihood estimation

    Args:
        UL: The number of unspliced labeled mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        SL: The number of spliced labeled mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The initial value of gamma. shape: (n_var,).
        beta_init: The initial value of beta. shape: (n_var,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).
        Total_smoothed: The number of total mRNA expression after normalization and smoothing for each gene in each cell. shape: (n_var, n_obs).
        S_smoothed: The number of spliced mRNA expression after normalization and smoothing for each gene in each cell. shape: (n_var, n_obs).

    Returns:
        gamma_s: The estimated spliced mRNA degradation rate gamma_s. shape: (n_var,).
        select_genes: Genes selected according to R2. shape: (n_var,).
        beta: The estimated gene specific splicing rate beta. shape: (n_var,).
        gamma_t: The estimated total mRNA degradation rate gamma_t. shape: (n_var,).
        gamma_r2: The R2 of gamma without correction. shape: (n_var,).
        alpha: The estimated gene specific transcription rate alpha. shape: (n_var,).
    """
    n_var = UL.shape[0]
    n_obs = UL.shape[1]
    gamma_s = np.zeros(n_var)
    select_genes = np.zeros(n_var, dtype=bool)
    gamma_r2 = np.zeros(n_var)
    beta = np.zeros(n_var)
    alpha = np.zeros(n_var)
    gamma_t = np.zeros(n_var)

    for i, ul, sl, r, s in tqdm(
            zip(np.arange(n_var), UL, SL, Total_smoothed, S_smoothed),
            "Estimate gamma via maximum likelihood estimation based on the ICSP model "
    ):
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()
        r = r.A.flatten() if issparse(r) else r.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()

        cell_capture_rate = cell_total / np.median(cell_total)

        def loss_func(parameters):
            # Loss function of independent cell specific Poisson model
            parameter_alpha, parameter_beta, parameter_gamma_s = parameters
            mu_u = parameter_alpha / parameter_beta * (1 - np.exp(-parameter_beta * time)) * cell_capture_rate
            mu_s = (parameter_alpha / parameter_gamma_s * (1 - np.exp(-parameter_gamma_s * time)) + parameter_alpha /
                    (parameter_gamma_s - parameter_beta) * (np.exp(-parameter_gamma_s * time) - np.exp(
                        -parameter_beta * time))) * cell_capture_rate
            loss_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss = loss_u + loss_s
            return loss

        # The initial values of gamma_s, beta and alpha are obtained from the initial values of gamma_t.
        gamma_s_init = gamma_init[i] * np.sum(r * s) / np.sum(np.power(s, 2))
        beta_init_new = beta_init[i] * gamma_s_init / gamma_init[i]
        alpha_init = np.mean(ul + sl) / np.mean(cell_capture_rate * (
                (1 - np.exp(-beta_init_new * time)) / beta_init_new + (1 - np.exp(-gamma_s_init * time)) / gamma_s_init
                + (np.exp(-gamma_s_init * time) - np.exp(-beta_init_new * time)) / (gamma_s_init - beta_init_new)))

        # Initialize and add boundary conditions
        b1 = (0, None)
        b2 = (0, None)
        b3 = (0, None)
        bnds = (b1, b2, b3)

        parameters_init = np.array(
            [np.maximum(alpha_init, 0), np.maximum(beta_init_new, 0), np.maximum(gamma_s_init, 0)])

        # Solve
        res = minimize(loss_func, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alpha[i], beta[i], gamma_s[i] = parameters
        if not success:
            print(res.message)

        # Calculate deviance R2 as goodness of fit

        def null_loss_func(parameters_null):
            # Loss function of null model
            parameters_a0, parameters_b0 = parameters_null
            mu_u = parameters_a0 * cell_capture_rate
            mu_s = parameters_b0 * cell_capture_rate
            loss0_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss0_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss0 = loss0_u + loss0_s
            return loss0

        b0 = np.mean(ul) / np.mean(cell_capture_rate)
        c0 = np.mean(sl) / np.mean(cell_capture_rate)
        loss0 = null_loss_func((b0, c0))

        def saturated_loss_func():
            # Loss function of saturated model
            loss_saturated_u = -np.sum(ul[ul > 0] * np.log(ul[ul > 0]) - ul[ul > 0])
            loss_saturated_s = -np.sum(sl[sl > 0] * np.log(sl[sl > 0]) - sl[sl > 0])
            loss_saturated = loss_saturated_u + loss_saturated_s
            return loss_saturated

        loss_saturated = saturated_loss_func()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2[i] = 1 - (devanice / (2 * n_obs - 3)) / (null_devanice / (2 * n_obs - 2))  # + 0.82

        gamma_t[i] = gamma_s[i] * np.sum(np.power(s, 2)) / np.sum(r * s)

    # Top 40% genes were selected by goodness of fit
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma_s < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    select_genes[sort_index[:number_selected_genes]] = 1
    select_genes[sort_index[number_selected_genes + 1:]] = 0

    return gamma_s, select_genes, beta, gamma_t, gamma_r2, alpha

def MLE_ICSP_Without_SS(
        UL: Union[np.ndarray, csr_matrix],
        SL: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        cell_total: np.ndarray,
        scv_gamma,
        scv_beta,
        U_smoothed,
        S_smoothed,
        gamma_init,
        scv_t_switch,
        scv_time
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_var = UL.shape[0]
    n_obs = UL.shape[1]
    cell_capture_rate_source = cell_total / np.median(cell_total)

    # When there is only one labeling duration we can obtain the analytical solution directly but cannot define the
    # goodness-of-fit.
    gamma_s = np.zeros(n_var)*np.nan
    alpha = np.zeros(n_var)
    beta = np.zeros(n_var)*np.nan
    select_genes = np.zeros(n_var, dtype=bool)
    gamma_t = np.zeros(n_var)
    for i, ul, sl, scv_gamma_i, scv_beta_i, u, s, gamma_init_i, scv_t_switch_i, scv_time_i in tqdm(
            zip(np.arange(n_var), UL, SL, scv_gamma, scv_beta, U_smoothed, S_smoothed, gamma_init, scv_t_switch,
                scv_time),
            "Infer parameters via maximum likelihood estimation based on the ICSP model and scVelo"
    ):
        if np.isnan(scv_gamma_i) or np.isnan(gamma_init_i):
            continue
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()
        scv_time_i = scv_time_i.A.flatten() if issparse(scv_time_i) else scv_time_i.flatten()
        r = u + s
        t_unique = np.unique(time)

        eind_temp = find_extreme(r, r, perc_left=None, perc_right=30)
        # if np.sum(~eind_temp) == 0:
        #     eind_temp = find_extreme(r, r, perc_left=perc_left, perc_right=15)
        if np.sum(~eind_temp) == 0:
            eind_temp = np.zeros_like(eind_temp, dtype=bool)
        cell_capture_rate = cell_capture_rate_source[~eind_temp]
        r = r[~eind_temp]
        ul = ul[~eind_temp]
        sl = sl[~eind_temp]
        u = u[~eind_temp]
        s = s[~eind_temp]
        scv_time_i = scv_time_i[~eind_temp]

        # on_state = (ul+sl) > 0
        # on_state = (scv_beta_i * u - scv_gamma_i * s) > 0
        on_state = scv_time_i < scv_t_switch_i
        # on_state = np.logical_or((scv_beta_i * u - scv_gamma_i * s) > 0, (ul+sl) > 0)
        # on_state = np.logical_and((scv_beta_i * u - scv_gamma_i * s) > 0, (ul + sl) > 0)
        # on_state = np.ones_like(scv_gamma_i, dtype=bool)

        beta_div_gamma = scv_beta_i / scv_gamma_i

        def solve_gamma_func(gamma_s_temp_exp):
            # Equation for solving kappa
            return np.mean(sl[on_state]) / np.mean(ul[on_state]) - (
                    1 - np.exp(-np.absolute(gamma_s_temp_exp) * t_unique)) * beta_div_gamma \
                   / (1 - np.exp(-np.absolute(gamma_s_temp_exp) * beta_div_gamma * t_unique)) - beta_div_gamma / (
                           1 - beta_div_gamma) * \
                   (np.exp(-np.absolute(gamma_s_temp_exp) * t_unique) - np.exp(
                       -np.absolute(gamma_s_temp_exp) * beta_div_gamma * t_unique)) / \
                   (1 - np.exp(-np.absolute(gamma_s_temp_exp) * beta_div_gamma * t_unique))


        gamma_s_init_i = gamma_init_i * np.sum((u + s) * s) / np.sum(np.power(s, 2))
        if gamma_s_init_i <= 0:
            gamma_s_init_i = 0.5

        gamma_solve = root(solve_gamma_func, np.absolute(gamma_s_init_i))
        if np.absolute(solve_gamma_func(np.absolute(gamma_solve.x))[0]) < 1e-4:
            gamma_s[i] = np.absolute(gamma_solve.x)
            beta[i] = np.absolute(gamma_solve.x) * beta_div_gamma
            select_genes[i] = 1
        else:
            select_genes[i] = 0

        alpha[i] = np.mean(ul[on_state]) / np.mean(cell_capture_rate[on_state]) * beta[i] / (
                1 - np.exp(-beta[i] * t_unique))
        gamma_t[i] = gamma_s[i] * np.sum(np.power(s, 2)) / np.sum(r * s)

    return alpha, beta, gamma_s, select_genes, gamma_t


def Cell_Specific_Alpha_Beta(
        UL_smoothed_CSP: Union[np.ndarray, csr_matrix],
        SL_smoothed_CSP: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        gamma_init: np.ndarray,
        beta_init: np.ndarray,
) -> Tuple[csr_matrix, csr_matrix]:
    """"Infer cell specific transcription rate and splicing rate based on ICSP model

    Args:
        UL_smoothed_CSP: The number of unspliced labeled mRNA expression after smoothing based on CSP type model for
        each gene in each cell. shape: (n_var, n_obs).
        SL_smoothed_CSP: The number of spliced labeled mRNA expression after smoothing based on CSP type model for
        each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        gamma_init: The gene wise initial value of gamma. shape: (n_var,).
        beta_init: The gene wise initial value of beta. shape: (n_var,).

    Returns: alpha_cs, beta_cs
        alpha_cs: The transcription rate for each gene in each cell. shape: (n_var, n_obs).
        beta_cs: The splicing rate for each gene in each cell. shape: (n_var, n_obs).
    """
    beta_cs = np.zeros_like(UL_smoothed_CSP.A) if issparse(UL_smoothed_CSP) else np.zeros_like(UL_smoothed_CSP)

    n_var = UL_smoothed_CSP.shape[0]
    n_obs = UL_smoothed_CSP.shape[1]

    for i, ul, sl, gamma_i, beta_i in tqdm(
            zip(np.arange(n_var), UL_smoothed_CSP, SL_smoothed_CSP, gamma_init, beta_init),
            "Estimate cell specific alpha and beta"
    ):
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()

        for j in range(n_obs):
            sl_j = sl[j]
            ul_j = ul[j]
            sl_div_ul_j = sl_j / ul_j
            time_j = time[j]

            def solve_beta_func(beta_j):
                # Equation for solving cell specific beta
                return sl_div_ul_j - (1 - np.exp(-gamma_i * time_j)) / gamma_i * beta_j / (1 - np.exp(-beta_j * time_j)) \
                       - beta_j / (gamma_i - beta_j) * (np.exp(-gamma_i * time_j) - np.exp(-beta_j * time_j)) / \
                       (1 - np.exp(-beta_j * time_j))

            beta_j_solve = root(solve_beta_func, beta_i)
            # beta_j_solve = fsolve(solve_beta_func, beta_i)

            beta_cs[i, j] = beta_j_solve.x

    k = 1 - np.exp(-beta_cs * (np.tile(time, (n_var, 1))))
    beta_cs = csr_matrix(beta_cs)
    alpha_cs = beta_cs.multiply(UL_smoothed_CSP).multiply(1 / k)
    return alpha_cs, beta_cs

def Select_SCV_Genes(adata):
    Ms = adata.layers['Ms'].T
    Mu = adata.layers['Mu'].T
    scv_gamma = adata.var.fit_gamma.values
    scv_beta = adata.var.fit_beta.values
    scv_alpha = adata.var.fit_alpha.values
    scv_t_switch = adata.var.fit_t_.values
    scv_time = adata.layers['fit_t'].T
    n_var = adata.n_vars
    var_name = adata.var_names

    def u_pred(t_obs, alpha, beta, gamma, t_switch):
        u_switch = alpha / beta * (1 - np.exp(-beta * t_switch))
        u_predict = np.zeros_like(t_obs)
        u_predict[t_obs < t_switch] = alpha / beta * (1 - np.exp(-beta * t_obs[t_obs < t_switch]))
        u_predict[t_obs >= t_switch] = u_switch * np.exp(-beta * (t_obs[t_obs >= t_switch] - t_switch))
        return u_predict

    def s_pred(t_obs, alpha, beta, gamma, t_switch):
        u_switch = alpha / beta * (1 - np.exp(-beta * t_switch))
        s_switch = alpha / gamma * (1 - np.exp(-gamma * t_switch)) + alpha / (gamma - beta) * (
                np.exp(-gamma * t_switch) - np.exp(-beta * t_switch))
        s_predict = np.zeros_like(t_obs)
        s_predict[t_obs < t_switch] = alpha / gamma * (1 - np.exp(-gamma * t_obs[t_obs < t_switch])) + alpha / (
                gamma - beta) * (np.exp(-gamma * t_obs[t_obs < t_switch]) - np.exp(-beta * t_obs[t_obs < t_switch]))
        s_predict[t_obs >= t_switch] = s_switch * np.exp(
            -gamma * (t_obs[t_obs >= t_switch] - t_switch)) - beta * u_switch * (
                                               np.exp(-gamma * (t_obs[t_obs >= t_switch] - t_switch)) - np.exp(
                                           -beta * (t_obs[t_obs >= t_switch] -
                                                    t_switch))) / (gamma - beta)
        return s_predict

    no_linear_r2 = np.zeros(n_var)
    for i, u, s, scv_gamma_i, scv_alpha_i, scv_beta_i, scv_t_switch_i, scv_time_i in tqdm(
            zip(np.arange(n_var), Mu, Ms, scv_gamma, scv_alpha, scv_beta, scv_t_switch, scv_time),
            "select scvelo gene"
    ):
        if np.isnan(scv_gamma_i):
            no_linear_r2[i] = -1000
            continue
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()
        scv_time_i = scv_time_i.A.flatten() if issparse(scv_time_i) else scv_time_i.flatten()

        u_bottom_10 = np.max(u) / 5
        s_bottom_10 = np.max(s) / 5
        index = np.logical_and(u > u_bottom_10, s > s_bottom_10)

        if np.sum(index)/len(u) < 0.03:
            no_linear_r2[i] = -1000
            continue

        # u_mean = np.mean(u)
        # s_mean = np.mean(s)

        u_mean = np.mean(u[index])
        s_mean = np.mean(s[index])

        u_predict = u_pred(scv_time_i, scv_alpha_i, scv_beta_i, scv_gamma_i, scv_t_switch_i)
        s_predict = s_pred(scv_time_i, scv_alpha_i, scv_beta_i, scv_gamma_i, scv_t_switch_i)


        no_linear_r2[i] = 1 - (
                    np.sum(np.square(u[index] - u_predict[index])) + np.sum(np.square(s[index] - s_predict[index]))) / (
                                  np.sum(np.square(u[index] - u_mean)) + np.sum(np.square(s[index] - s_mean)))

        if np.isnan(no_linear_r2[i]):
            no_linear_r2[i] = -1000

    adata.var['no_linear_r2'] = no_linear_r2

    return adata

def storm_sto_knn(
    adata: AnnData,
    X_data: Optional[np.ndarray] = None,
    genes: Optional[list] = None,
    group: Optional[str] = None,
    conn: Optional[csr_matrix] = None,
    use_gaussian_kernel: bool = False,
    normalize: bool = True,
    use_mnn: bool = False,
    layers: Union[List[str], str] = "all",
    n_pca_components: int = 30,
    n_neighbors: int = 30,
    copy: bool = False,
) -> Optional[AnnData]:
    """Calculate kNN based first and second moments (including uncentered covariance) for different layers of data.

    Args:
        adata: An AnnData object.
        X_data: The user supplied data that will be used for constructing the nearest neighbor graph directly. Defaults
            to None.
        genes: The one-dimensional numpy array of the genes that you want to perform pca analysis (if adata.obsm['X'] is
             not available). `X` keyname (instead of `X_pca`) was used to enable you use a different set of genes for
             flexible connectivity graph construction. If `None`, by default it will select genes based `use_for_pca`
             key in .var attributes if it exists otherwise it will also all genes stored in adata.X. Defaults to None.
        group: The column key/name that identifies the grouping information (for example, clusters that correspond to
            different cell types or different time points) of cells. This will be used to compute kNN graph for each
            group (i.e. cell-type/time-point). This is important, for example, we don't want cells from different
            labeling time points to be mixed when performing the kNN graph for calculating the moments. Defaults to
            None.
        conn: The connectivity graph that will be used for moment calculations. Defaults to None.
        use_gaussian_kernel: Whether to normalize the kNN graph via a Gaussian kernel. Defaults to False.
        normalize: Whether to normalize the connectivity matrix so that each row sums up to 1. When
            `use_gaussian_kernel` is False, this will be reset to be False because we will already normalize the
            connectivity matrix by dividing each row the total number of connections. Defaults to True.
        use_mnn: Whether to use mutual kNN across different layers as for the moment calculation. Defaults to False.
        layers: The layers that will be used for calculating the moments. Defaults to "all".
        n_pca_components: The number of pca components to use for constructing nearest neighbor graph and calculating
            1/2-st moments. Defaults to 30.
        n_neighbors: The number of pca components to use for constructing nearest neighbor graph and calculating 1/2-st
            moments. Defaults to 30.
        copy: Whether to return a new updated AnnData object or update inplace. Defaults to False.

    Raises:
        Exception: `group` is invalid.
        ValueError: `conn` is invalid. It should be a square array with dimension equal to the cell number.

    Returns:
        The updated AnnData object if `copy` is true. Otherwise, the AnnData object passed in would be updated inplace
        and None would be returned.
    """
    from dynamo.configuration import DKM, DynamoAdataKeyManager
    from dynamo.dynamo_logger import LoggerManager
    from dynamo.preprocessing.normalization import normalize_mat_monocle, sz_util
    from dynamo.preprocessing.pca import pca
    from dynamo.utils import copy_adata
    from dynamo.tools.connectivity import mnn, normalize_knn_graph, umap_conn_indices_dist_embedding
    from dynamo.tools.utils import elem_prod, get_mapper, inverse_norm
    from dynamo.tools.moments import gaussian_kernel, calc_1nd_moment


    adata = copy_adata(adata) if copy else adata
    mapper = get_mapper()

    if conn is None:
        if genes is None and "use_for_pca" in adata.var.keys():
            genes = adata.var_names[adata.var.use_for_pca]
        if use_mnn:
            if "mnn" not in adata.uns.keys():
                adata = mnn(
                    adata,
                    n_pca_components=n_pca_components,
                    layers="all",
                    use_pca_fit=True,
                    save_all_to_adata=False,
                )
            conn = adata.uns["mnn"]
        else:
            if X_data is not None:
                X = X_data
            else:
                if DKM.X_PCA not in adata.obsm.keys():
                    if not any([i.startswith("X_") for i in adata.layers.keys()]):
                        from dynamo.preprocessing import Preprocessor

                        genes_to_use = adata.var_names[genes] if genes.dtype == "bool" else genes
                        preprocessor = Preprocessor(force_gene_list=genes_to_use)
                        preprocessor.preprocess_adata(adata, recipe="monocle")
                    else:
                        CM = adata.X if genes is None else adata[:, genes].X
                        cm_genesums = CM.sum(axis=0)
                        valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
                        valid_ind = np.array(valid_ind).flatten()
                        CM = CM[:, valid_ind]
                        adata, fit, _ = pca(
                            adata,
                            CM,
                            n_pca_components=n_pca_components,
                            return_all=True,
                        )

                        adata.uns["explained_variance_ratio_"] = fit.explained_variance_ratio_[1:]

                X = adata.obsm[DKM.X_PCA][:, :n_pca_components]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if group is None:
                    (kNN, knn_indices, knn_dists, _,) = umap_conn_indices_dist_embedding(
                        X,
                        n_neighbors=np.min((n_neighbors, adata.n_obs - 1)),
                        return_mapper=False,
                    )

                    if use_gaussian_kernel and not use_mnn:
                        conn = gaussian_kernel(X, knn_indices, sigma=10, k=None, dists=knn_dists)
                    else:
                        conn = normalize_knn_graph(kNN > 0)
                        normalize = False
                else:
                    if group not in adata.obs.keys():
                        raise Exception(f"the group {group} provided is not a column name in .obs attribute.")
                    conn = csr_matrix((adata.n_obs, adata.n_obs))
                    cells_group = adata.obs[group]
                    uniq_grp = np.unique(cells_group)
                    for cur_grp in uniq_grp:
                        cur_cells = cells_group == cur_grp
                        cur_X = X[cur_cells, :]
                        (cur_kNN, cur_knn_indices, cur_knn_dists, _,) = umap_conn_indices_dist_embedding(
                            cur_X,
                            n_neighbors=np.min((n_neighbors, sum(cur_cells) - 1)),
                            return_mapper=False,
                        )

                        if use_gaussian_kernel and not use_mnn:
                            cur_conn = gaussian_kernel(
                                cur_X,
                                cur_knn_indices,
                                sigma=10,
                                k=None,
                                dists=cur_knn_dists,
                            )
                        else:
                            cur_conn = normalize_knn_graph(cur_kNN > 0)

                        cur_cells_ = np.where(cur_cells)[0]
                        conn[cur_cells_[:, None], cur_cells_] = cur_conn
    else:
        if conn.shape[0] != conn.shape[1] or conn.shape[0] != adata.n_obs:
            raise ValueError(
                "The connectivity data `conn` you provided should a square array with dimension equal to "
                "the cell number!"
            )

    layers = DynamoAdataKeyManager.get_available_layer_keys(adata, layers, False, False)

    # for CSP-type method
    layers_raw = [
        layer
        for layer in layers
        if not(layer.startswith("X")) and not(layer.startswith("M")) and (not layer.endswith("matrix") and not layer.endswith("ambiguous") and not(layer.startswith("fit")))
    ]
    layers_raw.sort(reverse=True)  # ensure we get M_CSP_us, M_CSP_tn, etc (instead of M_CSP_su or M_CSP_nt).

    # for CSP-type method
    size_factor = adata.obs['Size_Factor'].astype("float").values
    mapper_CSP = {
        "new": "M_CSP_n",
        "old": "M_CSP_o",
        "total": "M_CSP_t",
        "uu": "M_CSP_uu",
        "ul": "M_CSP_ul",
        "su": "M_CSP_su",
        "sl": "M_CSP_sl",
        "unspliced": "M_CSP_u",
        "spliced": "M_CSP_s",
    }

    # for CSP-type method
    for i, layer in enumerate(layers_raw):
        layer_x = adata.layers[layer].copy()
        layer_x = inverse_norm(adata, layer_x)

        if mapper_CSP[layer] not in adata.layers.keys():
            local_size_factor = conn.dot(size_factor)
            local_raw_counts = conn.dot(layer_x)
            adata.layers[mapper_CSP[layer]] = csr_matrix(local_raw_counts/local_size_factor.reshape(-1,1))

    adata.obsp["moments_con"] = conn

    if copy:
        return adata
    return None