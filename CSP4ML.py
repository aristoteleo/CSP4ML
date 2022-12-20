#!/usr/bin/env python 
# -*- coding:utf-8 -*-

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

def MLE_Cell_Specific_Poisson(N, time, gamma_init, cell_total):
    """"Infer parameters based on cell specific Poisson distributions using maximum likelihood estimation"""
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    alpha = np.zeros(n_var)
    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Infer parameters via Maximum Likelihood Estimation"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def Loss(parameters):
            # cell specific Poisson loss function
            parameter_alpha_div_gamma, parameter_gamma = parameters
            mu = parameter_alpha_div_gamma * (1 - np.exp(-parameter_gamma * time)) * cell_capture_rate
            loss = -np.sum(n * np.log(mu) - mu)
            return loss

        # Initialize and add boundary conditions
        alpha_div_gamma_init = np.mean(n) / np.mean(cell_capture_rate * (1 - np.exp(-gamma_init[i] * time)))
        b1 = (0, 10 * alpha_div_gamma_init)
        b2 = (0, 10 * gamma_init[i])
        bnds = (b1, b2)
        parameters_init = np.array([alpha_div_gamma_init, gamma_init[i]])

        # Solve
        res = minimize(Loss, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alpha[i], gamma[i] = parameters
        if not success:
            print(res.message)

        # Deviance R2 was calculated as goodness of fit
        def Null_Loss(parameters_null):
            parameters_a0 = parameters_null
            mu = parameters_a0 * cell_capture_rate
            loss0 = -np.sum(n * np.log(mu) - mu)
            return loss0

        a0 = np.mean(n) / np.mean(cell_capture_rate)
        loss0 = Null_Loss(a0)

        def Saturated_log_likehood():
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        loss_saturated = Saturated_log_likehood()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))  # + 0.83  # 0.467

    # Top40% genes were selected by goodness of fit
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    gamma_r2[sort_index[:number_selected_genes]] = 1
    gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma, gamma_r2

def MLE_Cell_Specific_Zero_Inflated_Poisson(N, time, gamma_init, cell_total):
    """"Infer parameters based on cell specific zero-inflated Poisson distributions using maximum likelihood estimation"""
    n_var = N.shape[0]
    n_obs = N.shape[1]
    gamma = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    prob_off = np.zeros(n_var)
    alphadivgamma = np.zeros(n_var)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Infer parameters via Maximum Likelihood Estimation"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        def Loss(parameters):
            # cell specific zero-inflated Poisson loss function
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
                                temp * temp * (s2_n - mean_n))
        alphadivgamma_init = mean_n / ((1 - prob_off_init) * temp)
        b1 = (0, 10 * alphadivgamma_init)
        b2 = (0, 10 * gamma_init[i])
        b3 = (0, (np.sum(n < 0.001) / np.sum(n > -1)))
        bnds = (b1, b2, b3)
        parameters_init = np.array([alphadivgamma_init, gamma_init[i], prob_off_init])

        # Slove
        res = minimize(Loss, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        alphadivgamma[i], gamma[i], prob_off[i] = parameters
        loss = res.fun
        success = res.success
        if not success:
            print(res.message)

        # Deviance R2 was calculated as goodness of fit
        def Null_Loss(parameters_null):
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
        res_null = minimize(Null_Loss, parameters_init_null, method='SLSQP', bounds=bnds_null, tol=1e-2,
                            options={'maxiter': 1000})
        loss0 = res_null.fun

        def Saturated_log_likehood():
            loss_saturated = -np.sum(n[n > 0] * np.log(n[n > 0]) - n[n > 0])
            return loss_saturated

        loss_saturated = Saturated_log_likehood()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)

        gamma_r2[i] = 1 - (devanice / (n_obs - 2)) / (null_devanice / (n_obs - 1))  # + 0.467 # + 0.83

        # Top40% genes were selected by goodness of fit
        number_selected_genes = int(n_var * 0.4)
        gamma_r2[gamma < 0.01] = 0
        sort_index = np.argsort(-gamma_r2)
        gamma_r2[sort_index[:number_selected_genes]] = 1
        gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma, gamma_r2

def MLE_Independent_Cell_Specific_Poisson(UL, SL, time, gamma_init, beta_init, cell_total, Total_smoothed, S_smoothed):
    """"Infer parameters based on independent cell specific Poisson distributions using maximum likelihood estimation"""
    n_var = UL.shape[0]
    n_obs = UL.shape[1]
    gamma_s = np.zeros(n_var)
    gamma_r2 = np.zeros(n_var)
    beta = np.zeros(n_var)
    alpha = np.zeros(n_var)
    gamma_t = np.zeros(n_var)

    for i, ul, sl, r, s in tqdm(
            zip(np.arange(n_var), UL, SL, Total_smoothed, S_smoothed),
            "Estimate gamma via Maximum Likelihood Estimation"
    ):
        sl = sl.A.flatten() if issparse(sl) else sl.flatten()
        ul = ul.A.flatten() if issparse(ul) else ul.flatten()
        r = r.A.flatten() if issparse(r) else r.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()

        cell_capture_rate = cell_total / np.median(cell_total)

        def Loss(parameters):
            # loss function
            parameter_alpha, parameter_beta, parameter_gamma_s = parameters
            mu_u = parameter_alpha / parameter_beta * (1 - np.exp(-parameter_beta * time)) * cell_capture_rate
            mu_s = (parameter_alpha / parameter_gamma_s * (1 - np.exp(-parameter_gamma_s * time)) + parameter_alpha /
                    (parameter_gamma_s - parameter_beta) * (np.exp(-parameter_gamma_s * time) - np.exp(
                        -parameter_beta * time))) * cell_capture_rate
            loss_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss = loss_u + loss_s
            return loss

        gamma_s_init = gamma_init[i] * np.sum(r * s) / np.sum(np.power(s, 2))
        beta_init_new = beta_init[i] * gamma_s_init / gamma_init[i]
        alpha_init = np.mean(ul + sl) / np.mean(cell_capture_rate * ((1 - np.exp(-beta_init_new * time)) / beta_init_new
                   + (1 - np.exp(-gamma_s_init * time)) / gamma_s_init + (np.exp(-gamma_s_init * time) - np.exp(
                    -beta_init_new * time)) / (gamma_s_init - beta_init_new)))

        # Initialize and add boundary conditions
        b1 = (0, 10 * alpha_init)
        b2 = (0, 10 * beta_init_new)
        b3 = (0, 10 * gamma_s_init)
        bnds = (b1, b2, b3)
        parameters_init = np.array([alpha_init, beta_init_new, gamma_s_init])

        # Solve
        res = minimize(Loss, parameters_init, method='SLSQP', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='Nelder-Mead', tol=1e-2, options={'maxiter': 1000})
        # res = minimize(Loss, parameters_init, method='COBYLA', bounds=bnds, tol=1e-2, options={'maxiter': 1000})
        parameters = res.x
        loss = res.fun
        success = res.success
        alpha[i], beta[i], gamma_s[i] = parameters
        if not success:
            print(res.message)

        # Deviance R2 was calculated as goodness of fit
        def Null_Loss(parameters_null):
            parameters_a0, parameters_b0 = parameters_null
            mu_u = parameters_a0 * cell_capture_rate
            mu_s = parameters_b0 * cell_capture_rate
            loss0_u = -np.sum(ul * np.log(mu_u) - mu_u)
            loss0_s = -np.sum(sl * np.log(mu_s) - mu_s)
            loss0 = loss0_u + loss0_s
            return loss0

        b0 = np.mean(ul) / np.mean(cell_capture_rate)
        c0 = np.mean(sl) / np.mean(cell_capture_rate)
        loss0 = Null_Loss((b0, c0))

        def saturated_log_likehood():
            loss_saturated_u = -np.sum(ul[ul > 0] * np.log(ul[ul > 0]) - ul[ul > 0])
            loss_saturated_s = -np.sum(sl[sl > 0] * np.log(sl[sl > 0]) - sl[sl > 0])
            loss_saturated = loss_saturated_u + loss_saturated_s
            return loss_saturated

        loss_saturated = saturated_log_likehood()
        null_devanice = 2 * (loss0 - loss_saturated)
        devanice = 2 * (loss - loss_saturated)
        gamma_r2[i] = 1 - (devanice / (2 * n_obs - 3)) / (null_devanice / (2 * n_obs - 2))  # + 0.82

        gamma_t[i] = gamma_s[i] * np.sum(np.power(s, 2)) / np.sum(r * s)

    # Top40% genes were selected by goodness of fit
    number_selected_genes = int(n_var * 0.4)
    gamma_r2[gamma_s < 0.01] = 0
    sort_index = np.argsort(-gamma_r2)
    gamma_r2[sort_index[:number_selected_genes]] = 1
    gamma_r2[sort_index[number_selected_genes + 1:]] = 0

    return gamma_s, gamma_r2, beta, gamma_t

def cell_specific_alpha_beta(UL, SL, time, gamma, beta):
    """"Inferring Cell-Specific Transcription Rate and Splicing Rate"""
    beta_cs = np.zeros_like(UL.A) if issparse(UL) else np.zeros_like(UL)

    n_var = UL.shape[0]
    n_obs = UL.shape[1]

    for i, ul, sl, gamma_i, beta_i in tqdm(
            zip(np.arange(n_var), UL, SL, gamma, beta),
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
                return sl_div_ul_j - (1 - np.exp(-gamma_i * time_j)) / gamma_i * beta_j / (
                            1 - np.exp(-beta_j * time_j)) - beta_j / (gamma_i - beta_j) * (
                                   np.exp(-gamma_i * time_j) - np.exp(-beta_j * time_j)) / (
                                   1 - np.exp(-beta_j * time_j))

            beta_j_solve = root(solve_beta_func, beta_i)
            # beta_j_solve = fsolve(solve_beta_func, beta_i)

            beta_cs[i, j] = beta_j_solve.x

    k = 1 - np.exp(-beta_cs*(np.tile(time, (n_var, 1))))
    beta_cs = csr_matrix(beta_cs)
    alpha_cs = beta_cs.multiply(UL).multiply(1 / k)
    return alpha_cs, beta_cs