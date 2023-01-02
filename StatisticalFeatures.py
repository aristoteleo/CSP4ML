#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import issparse
from scipy.stats import chi2_contingency
from tqdm import tqdm
from scipy import stats

def Test_Same_Distribution_Total(R, time):
    """Test whether total RNA obeyed the same distribution via chi-square independence test"""
    n_var = R.shape[0]
    p_value = np.zeros(n_var)
    flag = np.zeros(n_var)
    for i, r in tqdm(
            zip(np.arange(n_var), R),
            "Test whether total RNA obeyed the same distribution via chi-square independence test"
    ):
        r = r.A.flatten() if issparse(r) else r.flatten()
        T = np.unique(time)

        num_class = np.int(np.max(r)) + 1
        chisqure_table = np.zeros((num_class, len(T)))

        range_max = num_class
        range_min = 0
        for j in range(len(T)):
            r_j = r[time == T[j]]
            f_obs_j = np.bincount(r_j.astype(np.int16), minlength=num_class)
            chisqure_table[:, j] = f_obs_j
            range_min_j = (f_obs_j.astype(np.int16) != 0).argmax()
            range_max_j = (f_obs_j[max(range_min_j, int(num_class / 10)):].astype(
                np.int16) == 0).argmax() + max(range_min_j, int(num_class / 10))
            if (range_max_j < range_max) and (range_max_j > max(range_min_j, int(num_class / 10))):
                range_max = range_max_j
            if range_min_j > range_min:
                range_min = range_min_j

        chisqure_table[range_min, :] = np.sum(chisqure_table[:range_min + 1, :], axis=0)
        chisqure_table[range_max - 1, :] = np.sum(chisqure_table[range_max - 1:, :], axis=0)
        chisqure_table = chisqure_table[range_min:range_max, :]

        while np.any(chisqure_table[-1, :] < 5):
            chisqure_table[-2, :] = chisqure_table[-2, :] + chisqure_table[-1, :]
            chisqure_table = chisqure_table[0:-1, :]

        adjust_number = 0
        while np.any(chisqure_table[adjust_number, :] < 5):
            chisqure_table[adjust_number + 1, :] = chisqure_table[adjust_number + 1, :] + chisqure_table[adjust_number,
                                                                                          :]
            adjust_number = adjust_number + 1
        chisqure_table = chisqure_table[adjust_number:, :]

        if np.any(chisqure_table == 0):
            chisqure_table_new = np.zeros_like(chisqure_table)
            row_number = 0
            for k in range(chisqure_table.shape[0]):
                temp = chisqure_table[k, :]
                while np.any(temp == 0):
                    k = k + 1
                    temp = temp + chisqure_table[k, :]
                chisqure_table_new[row_number, :] = temp
                row_number = row_number + 1
            if row_number == 1:
                flag[i] = 2
                break
            chisqure_table_new = chisqure_table_new[:row_number, :]
            _, p_value[i], _, _ = chi2_contingency(chisqure_table_new)
        else:
            _, p_value[i], _, _ = chi2_contingency(chisqure_table)

        if p_value[i] > 0.05:
            flag[i] = 1

    reject_rate = np.sum(flag == 0) / n_var
    accept_rate = np.sum(flag == 1) / n_var
    unable_to_determine_rate = np.sum(flag == 2) / n_var

    print("Test whether total RNA obeyed the same distribution")
    print("accept", accept_rate)
    print("reject", reject_rate)
    print("unable to determine", unable_to_determine_rate)

    return accept_rate, reject_rate, unable_to_determine_rate

def Test_CSP(N, time, cell_total):
    """Test whether the new RNA obeys the cell specific Poisson distribution"""
    n_var = N.shape[0]
    p_value = np.zeros((n_var, len(np.unique(time))))
    flag = np.zeros((n_var, len(np.unique(time))))
    skip_number = 0

    T = np.unique(time)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Test whether the new RNA obeys the cell specific Poisson distribution"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        for j in range(len(T)):
            n_j = n[time == T[j]]
            mean_j = np.mean(n_j)
            cell_capture_rate_j = cell_capture_rate[time == T[j]]
            mean_cell_capture_rate_j = np.mean(cell_capture_rate_j)
            parms_lambda = mean_j / mean_cell_capture_rate_j

            num_class = np.int(np.max(n_j)) + 2

            Prob_Matrix = np.zeros((len(n_j), num_class))
            for k in range(len(n_j)):
                Poisson = stats.poisson(mu=parms_lambda * cell_capture_rate_j[k])
                exp_prob = Poisson.pmf(range(num_class))
                exp_prob[-1] = 1 - Poisson.cdf(num_class - 1 - 0.01)
                Prob_Matrix[k, :] = exp_prob
            Prob_mean = np.mean(Prob_Matrix, axis=0)
            f_exp = Prob_mean * len(n_j)

            while f_exp[-1] < 0.25:
                Prob_Matrix[:, -2] = Prob_Matrix[:, -2] + Prob_Matrix[:, -1]
                Prob_Matrix = Prob_Matrix[:, 0:-1]
                Prob_mean[-2] = Prob_mean[-2] + Prob_mean[-1]
                Prob_mean = Prob_mean[0:-1]
                f_exp[-2] = f_exp[-2] + f_exp[-1]
                f_exp = f_exp[0:-1]

            num_class = len(f_exp)
            if num_class <= 2:
                skip_number = skip_number + 1
                flag[i, j] = 2
                continue

            D = len(n_j) * np.diag(Prob_mean)
            for k in range(len(n_j)):
                exp_prob = Prob_Matrix[k, :]
                temp = exp_prob.reshape(-1, 1)
                D = D - temp * np.transpose(temp)

            f_obs = np.bincount(n_j.astype(np.int16), minlength=num_class)
            f_obs[num_class - 1] = np.sum(f_obs[num_class - 1:])
            f_obs = f_obs[:num_class]

            f_exp_star = f_exp[0:-1]
            f_obs_star = f_obs[0:-1]
            D_star = D[0:-1, 0:-1]
            f_diff = f_obs_star - f_exp_star
            # f_diff = np.abs(f_obs_star - f_exp_star)-0.5  # Yates Continuity Correction
            f_diff = f_diff.reshape(-1, 1)
            chi_sq = np.dot(np.transpose(f_diff), np.linalg.solve(D_star, f_diff))
            chi_sq_dis = stats.chi2(df=len(f_exp) - 1 - 1)
            p_value[i, j] = 1 - chi_sq_dis.cdf(chi_sq)
            if p_value[i, j] > 0.05:
                flag[i, j] = 1

    reject_rate = np.sum(flag == 0, axis=0) / n_var
    accept_rate = np.sum(flag == 1, axis=0) / n_var
    unable_to_determine_rate = np.sum(flag == 2, axis=0) / n_var

    print("cell specific Poisson")
    print("time", T)
    print("accept", accept_rate)
    print("reject", reject_rate)
    print("unable to determine", unable_to_determine_rate)

    return accept_rate, reject_rate, unable_to_determine_rate


def Test_CSZIP(N, time, cell_total):
    """Test whether the new RNA obeys the cell specific zero-inflated Poisson distribution"""
    n_var = N.shape[0]
    p_value = np.zeros((n_var, len(np.unique(time))))
    flag = np.zeros((n_var, len(np.unique(time))))
    skip_number = 0
    T = np.unique(time)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Test whether the new RNA obeys the cell specific zero-inflated Poisson distribution"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        for j in range(len(T)):
            n_j = n[time == T[j]]
            cell_capture_rate_j = cell_capture_rate[time == T[j]]

            def zipossion_like_hood(params):
                params_lambda, params_zero_rate = params
                mu = params_lambda * cell_capture_rate_j
                n_eq_0_index = n_j < 0.0001
                n_over_0_index = n_j > 0.0001

                loglikehood_eq0 = np.sum(np.log(params_zero_rate + (1 - params_zero_rate) * np.exp(-mu[n_eq_0_index])))
                loglikehood_over0 = np.sum(
                    np.log(1 - params_zero_rate) + (-mu[n_over_0_index]) + n_j[n_over_0_index] * np.log(
                        mu[n_over_0_index]))

                negsumloglikehood = -(loglikehood_eq0 + loglikehood_over0)
                return negsumloglikehood

            mean_j = np.mean(n_j)
            s2_j = np.mean(np.power(n_j, 2))
            mean_cell_capture_rate_j = np.mean(cell_capture_rate_j)
            params_zero_rate_init = 1 - mean_j * mean_j * np.mean(np.power(cell_capture_rate_j, 2)) / (
                    mean_cell_capture_rate_j * mean_cell_capture_rate_j * (s2_j - mean_j))
            params_lambda_init = np.mean(n_j) / (mean_cell_capture_rate_j * (1 - params_zero_rate_init))
            params0 = np.array([params_lambda_init, params_zero_rate_init])

            b1 = (0, 100 * params_lambda_init)
            b2 = (0, np.sum(n_j < 0.001) / np.sum(n_j > -1))
            bnds = (b1, b2)
            res = minimize(zipossion_like_hood, params0, method='SLSQP', bounds=bnds)
            params = res.x
            params_lambda, params_zero_rate = params

            num_class = np.int(np.max(n_j)) + 2

            Prob_Matrix = np.zeros((len(n_j), num_class))
            for k in range(len(n_j)):
                Poisson = stats.poisson(mu=params_lambda * cell_capture_rate_j[k])
                exp_prob = Poisson.pmf(range(num_class))
                exp_prob[-1] = 1 - Poisson.cdf(num_class - 1 - 0.01)
                exp_prob = exp_prob * (1 - params_zero_rate)
                exp_prob[0] = exp_prob[0] + params_zero_rate
                Prob_Matrix[k, :] = exp_prob
            Prob_mean = np.mean(Prob_Matrix, axis=0)
            f_exp = Prob_mean * len(n_j)

            while f_exp[-1] < 0.25:
                Prob_Matrix[:, -2] = Prob_Matrix[:, -2] + Prob_Matrix[:, -1]
                Prob_Matrix = Prob_Matrix[:, 0:-1]
                Prob_mean[-2] = Prob_mean[-2] + Prob_mean[-1]
                Prob_mean = Prob_mean[0:-1]
                f_exp[-2] = f_exp[-2] + f_exp[-1]
                f_exp = f_exp[0:-1]

            num_class = len(f_exp)
            if num_class <= 3:
                skip_number = skip_number + 1
                flag[i, j] = 2
                continue

            D = len(n_j) * np.diag(Prob_mean)
            for k in range(len(n_j)):
                exp_prob = Prob_Matrix[k, :]
                temp = exp_prob.reshape(-1, 1)
                D = D - temp * np.transpose(temp)

            f_obs = np.bincount(n_j.astype(np.int16), minlength=num_class)
            f_obs[num_class - 1] = np.sum(f_obs[num_class - 1:])
            f_obs = f_obs[:num_class]


            f_exp_star = f_exp[0:-1]
            f_obs_star = f_obs[0:-1]
            D_star = D[0:-1, 0:-1]
            f_diff = f_obs_star - f_exp_star
            # f_diff = np.abs(f_obs_star - f_exp_star)-0.5 # Yates Continuity Correction
            f_diff = f_diff.reshape(-1, 1)
            chi_sq = np.dot(np.transpose(f_diff), np.linalg.solve(D_star, f_diff))
            chi_sq_dis = stats.chi2(df=len(f_exp) - 1 - 2)
            p_value[i, j] = 1 - chi_sq_dis.cdf(chi_sq)
            if p_value[i, j] > 0.05:
                flag[i, j] = 1

    reject_rate = np.sum(flag == 0, axis=0) / n_var
    accept_rate = np.sum(flag == 1, axis=0) / n_var
    unable_to_determine_rate = np.sum(flag == 2, axis=0) / n_var

    print("cell specific ZI Poisson")
    print("time", T)
    print("accept", accept_rate)
    print("reject", reject_rate)
    print("unable to determine", unable_to_determine_rate)

    return accept_rate, reject_rate, unable_to_determine_rate
