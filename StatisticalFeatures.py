from typing import Tuple, Union, Optional, List
from anndata import AnnData
from pandas import DataFrame, Series
from scipy.sparse import (
    csr_matrix,
    issparse,
    SparseEfficiencyWarning,
)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2_contingency
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt


def new_counts_statistical_features(
        adata: AnnData,
        gene_list: list,
        model: str = 'CSP',
        significance: float = 0.05
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Perform cell-specific Chi-square test for new mRNA raw counts of given genes using the specified model
    (CSP or CSZIP), and judge whether to accept or reject the null hypothesis based on the given significance level.

    Args:
        adata: class:`~anndata.AnnData`
            an Annodata object
        gene_list: A list of gene names that are going to be studied.
        model: Cell-specific stochastic model used ('CSP' or 'CSZIP', default is 'CSP').
        significance: Significance level of hypothesis test (default is 0.05).

    Returns:
        result_flag: A DataFrame that holds the results of the hypothesis test for each gene and each labeling duration.
         0 means the null hypothesis is rejected, 1 means the null hypothesis is accepted, and 2 means that the
         expression is too low to be determined.
        p_value: A DataFrame that holds the P-value of the hypothesis test for each gene and each labeling duration.
        f_obs: A DataFrame that holds the observed new mRNA counts for each gene and each labeling duration.
        f_exp: A DataFrame that holds the expected counts for the given model for each gene and each labeling duration.
    """
    subset_adata = adata[:, gene_list]
    time = subset_adata.obs['time']
    N = subset_adata.layers['new'].T
    cell_total = subset_adata.obs['initial_cell_size'].astype("float").values
    if model == 'CSP':
        result_flag, p_value, f_obs, f_exp = test_CSP(N, time, cell_total, significance)
    elif model == 'CSZIP':
        result_flag, p_value, f_obs, f_exp = test_CSZIP(N, time, cell_total, significance)

    T = np.unique(time).astype('str').tolist()
    result_flag = pd.DataFrame(result_flag, columns=T, index=gene_list)
    p_value = pd.DataFrame(p_value, columns=T, index=gene_list)
    f_obs = pd.DataFrame(f_obs, columns=T, index=gene_list)
    f_exp = pd.DataFrame(f_exp, columns=T, index=gene_list)

    return result_flag, p_value, f_obs, f_exp


def total_counts_statistical_features(
        adata: AnnData,
        gene_list: list,
        significance: float = 0.05
) -> Tuple[Series, Series, Series]:
    """Perform Chi-square independence test for total mRNA raw counts of given genes, and judge whether to accept or
    reject the null hypothesis based on the given significance level.

    Args:
        adata: class:`~anndata.AnnData`
            an Annodata object
        gene_list: A list of gene names that are going to be studied.
        significance: Significance level of hypothesis test (default is 0.05).

    Returns:
        result_flag: A Series that holds the results of the hypothesis test for each gene. 0 means the null hypothesis
        is rejected, 1 means the null hypothesis is accepted, and 2 means that it is unable to determine.
        p_value: A Series that holds the P-value of the hypothesis test for each gene.
        f_obs: A Series that holds the observed total mRNA counts for each gene.
    """
    subset_adata = adata[:, gene_list]
    time = subset_adata.obs['time']
    R = subset_adata.layers['total'].T
    result_flag, p_value, f_obs = test_same_distribution_total(R, time, significance)

    result_flag = pd.Series(result_flag, index=gene_list)
    p_value = pd.Series(p_value, index=gene_list)
    f_obs = pd.Series(f_obs, index=gene_list)

    return result_flag, p_value, f_obs


def test_same_distribution_total(
        R: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        significance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[DataFrame]]:
    """Test whether total mRNA counts obeyed the same distribution via chi-square independence test

    Args:
        R: The number of total mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        significance: Significance level of hypothesis test (default is 0.05).

    Returns:
        flag: The results of the hypothesis test for each gene. 0 means the null hypothesis is rejected, 1 means the
        null hypothesis is accepted, and 2 means that it is unable to determine. shape: (n_var,).
        p_value: The P-value of the hypothesis test for each gene. shape: (n_var,).
        f_obs_all: The observed total mRNA counts for each gene and each labeling durations. shape: (n_var,).
    """
    n_var = R.shape[0]
    p_value = np.zeros(n_var)
    flag = np.zeros(n_var)
    f_obs_all = []

    for i, r in tqdm(
            zip(np.arange(n_var), R),
            "Test whether total RNA counts obeyed the same distribution via chi-square independence test"
    ):
        r = r.A.flatten() if issparse(r) else r.flatten()
        T = np.unique(time)

        num_class = np.int(np.max(r)) + 1
        chisqure_table = np.zeros((num_class, len(T)))

        # Construct contingency table for chi-square independence test
        range_max = num_class
        range_min = 0
        for j in range(len(T)):
            r_j = r[time == T[j]]
            f_obs_j = np.bincount(r_j.astype(np.int16), minlength=num_class)
            chisqure_table[:, j] = f_obs_j

        # Combine intervals where the observed counts are too small
        while np.any(chisqure_table[-1, :] < 1):
            chisqure_table[-2, :] = chisqure_table[-2, :] + chisqure_table[-1, :]
            chisqure_table = chisqure_table[0:-1, :]
            range_max = range_max - 1

        adjust_number = 0
        while np.any(chisqure_table[adjust_number, :] < 1):
            chisqure_table[adjust_number + 1, :] = chisqure_table[adjust_number + 1, :] + chisqure_table[adjust_number,
                                                                                          :]
            adjust_number = adjust_number + 1
            range_min = range_min + 1
        chisqure_table = chisqure_table[adjust_number:, :]

        value_range = np.arange(range_min, range_max)
        value_all_zero_counts = np.all(chisqure_table == 0, axis=1)
        chisqure_table = chisqure_table[~value_all_zero_counts, :]
        value_range = value_range[~value_all_zero_counts]

        f_obs_all.append(pd.DataFrame(chisqure_table, columns=T, index=value_range))

        if len(value_range) == 1:
            flag[i] = 2
            continue

        # Chi-square independence test
        _, p_value[i], _, _ = chi2_contingency(chisqure_table)
        if p_value[i] > significance:
            flag[i] = 1

    return flag, p_value, f_obs_all


def test_CSP(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        cell_total: np.ndarray,
        significance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]], List[List[np.ndarray]]]:
    """Test whether the new mRNA counts obeys the cell specific Poisson distribution (CSP) model

    Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).
        significance: Significance level of hypothesis test (default is 0.05).

    Returns:
        flag: The results of the hypothesis test for each gene and each labeling duration. 0 means the null hypothesis
        is rejected, 1 means the null hypothesis is accepted, and 2 means that it is unable to determine.
        shape: (n_var,).
        p_value: The P-value of the hypothesis test for each gene and each labeling duration. shape: (n_var,).
        f_obs_all: The observed new mRNA counts for each gene and each labeling duration. shape: (n_var,).
        f_exp_all: The expected counts for the CSP model for each gene and each labeling duration. shape: (n_var,).
    """
    n_var = N.shape[0]
    p_value = np.zeros((n_var, len(np.unique(time))))
    flag = np.zeros((n_var, len(np.unique(time))))
    skip_number = 0
    f_obs_all = []
    f_exp_all = []

    T = np.unique(time)

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Test whether the new mRNA counts obeys the cell specific Poisson distribution"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        f_obs_i = []
        f_exp_i = []

        for j in range(len(T)):
            n_j = n[time == T[j]]
            mean_j = np.mean(n_j)
            cell_capture_rate_j = cell_capture_rate[time == T[j]]
            mean_cell_capture_rate_j = np.mean(cell_capture_rate_j)
            # Analytic solutions for the parameters of the maximum likelihood estimation
            parms_lambda = mean_j / mean_cell_capture_rate_j

            num_class = np.int(np.max(n_j)) + 2

            # Calculate expected counts for the CSP model
            Prob_Matrix = np.zeros((len(n_j), num_class))
            for k in range(len(n_j)):
                Poisson = stats.poisson(mu=parms_lambda * cell_capture_rate_j[k])
                exp_prob = Poisson.pmf(range(num_class))
                exp_prob[-1] = 1 - Poisson.cdf(num_class - 1 - 0.01)
                Prob_Matrix[k, :] = exp_prob
            Prob_mean = np.mean(Prob_Matrix, axis=0)
            f_exp = Prob_mean * len(n_j)

            # Combine intervals where the expected counts are too small
            while f_exp[-1] < 0.25:
                Prob_Matrix[:, -2] = Prob_Matrix[:, -2] + Prob_Matrix[:, -1]
                Prob_Matrix = Prob_Matrix[:, 0:-1]
                Prob_mean[-2] = Prob_mean[-2] + Prob_mean[-1]
                Prob_mean = Prob_mean[0:-1]
                f_exp[-2] = f_exp[-2] + f_exp[-1]
                f_exp = f_exp[0:-1]

            # Calculate observed counts
            num_class = len(f_exp)
            f_obs = np.bincount(n_j.astype(np.int16), minlength=num_class)
            f_obs[num_class - 1] = np.sum(f_obs[num_class - 1:])
            f_obs = f_obs[:num_class]

            f_exp_i.append(f_exp)
            f_obs_i.append(f_obs)

            # When the number of groupings is less than or equal to 2, the data will be fitted exactly because the
            # degrees of freedom are 0, and thus it cannot be determined whether it fits the CSP model.
            if num_class <= 2:
                skip_number = skip_number + 1
                flag[i, j] = 2
                continue

            # Cell-specific chi-square test for CSP model
            D = len(n_j) * np.diag(Prob_mean)
            for k in range(len(n_j)):
                exp_prob = Prob_Matrix[k, :]
                temp = exp_prob.reshape(-1, 1)
                D = D - temp * np.transpose(temp)

            f_exp_star = f_exp[0:-1]
            f_obs_star = f_obs[0:-1]
            D_star = D[0:-1, 0:-1]
            f_diff = f_obs_star - f_exp_star
            # f_diff = np.abs(f_obs_star - f_exp_star)-0.5  # Yates Continuity Correction
            f_diff = f_diff.reshape(-1, 1)
            chi_sq = np.dot(np.transpose(f_diff), np.linalg.solve(D_star, f_diff))
            chi_sq_dis = stats.chi2(df=len(f_exp) - 1 - 1)
            p_value[i, j] = 1 - chi_sq_dis.cdf(chi_sq)
            if p_value[i, j] > significance:
                flag[i, j] = 1

        f_obs_all.append(f_obs_i)
        f_exp_all.append(f_exp_i)

    return flag, p_value, f_obs_all, f_exp_all


def test_CSZIP(
        N: Union[np.ndarray, csr_matrix],
        time: np.ndarray,
        cell_total: np.ndarray,
        significance: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]], List[List[np.ndarray]]]:
    """Test whether the new RNA counts obeys the cell specific zero-inflated Poisson distribution

    Args:
        N: The number of new mRNA counts for each gene in each cell. shape: (n_var, n_obs).
        time: The time point of each cell. shape: (n_obs,).
        cell_total: The total counts of reads for each cell. shape: (n_obs,).
        significance: Significance level of hypothesis test (default is 0.05).

    Returns:
        flag: The results of the hypothesis test for each gene and each labeling duration. 0 means the null hypothesis
        is rejected, 1 means the null hypothesis is accepted, and 2 means that it is unable to determine.
        shape: (n_var,).
        p_value: The P-value of the hypothesis test for each gene and each labeling duration. shape: (n_var,).
        f_obs_all: The observed new mRNA counts for each gene and each labeling duration. shape: (n_var,).
        f_exp_all: The expected counts for the CSZIP model for each gene and each labeling duration. shape: (n_var,).
    """
    n_var = N.shape[0]
    p_value = np.zeros((n_var, len(np.unique(time))))
    flag = np.zeros((n_var, len(np.unique(time))))
    skip_number = 0
    T = np.unique(time)
    f_obs_all = []
    f_exp_all = []

    for i, n in tqdm(
            zip(np.arange(n_var), N),
            "Test whether the new RNA obeys the cell specific zero-inflated Poisson distribution"
    ):
        n = n.A.flatten() if issparse(n) else n.flatten()
        cell_capture_rate = cell_total / np.median(cell_total)

        f_obs_i = []
        f_exp_i = []
        for j in range(len(T)):
            n_j = n[time == T[j]]
            cell_capture_rate_j = cell_capture_rate[time == T[j]]

            def _cszip_neg_log_likehood(params):
                # CSZIP model negative log-likelihood function
                params_lambda, params_zero_rate = params
                mu = params_lambda * cell_capture_rate_j
                n_eq_0_index = n_j < 0.0001
                n_over_0_index = n_j > 0.0001

                loglikehood_eq0 = np.sum(np.log(params_zero_rate + (1 - params_zero_rate) * np.exp(-mu[n_eq_0_index])))
                loglikehood_over0 = np.sum(
                    np.log(1 - params_zero_rate) + (-mu[n_over_0_index]) + n_j[n_over_0_index] * np.log(
                        mu[n_over_0_index]))

                neg_sum_log_likehood = -(loglikehood_eq0 + loglikehood_over0)
                return neg_sum_log_likehood

            # Use moment estimation as initial value
            mean_j = np.mean(n_j)
            s2_j = np.mean(np.power(n_j, 2))
            mean_cell_capture_rate_j = np.mean(cell_capture_rate_j)
            params_zero_rate_init = 1 - mean_j * mean_j * np.mean(np.power(cell_capture_rate_j, 2)) / (
                    mean_cell_capture_rate_j * mean_cell_capture_rate_j * (s2_j - mean_j))
            params_lambda_init = np.mean(n_j) / (mean_cell_capture_rate_j * (1 - params_zero_rate_init))
            params0 = np.array([params_lambda_init, params_zero_rate_init])

            # Solve
            b1 = (0, 100 * params_lambda_init)
            b2 = (0, np.sum(n_j < 0.001) / np.sum(n_j > -1))
            bnds = (b1, b2)
            res = minimize(_cszip_neg_log_likehood, params0, method='SLSQP', bounds=bnds)
            params = res.x
            params_lambda, params_zero_rate = params

            num_class = np.int(np.max(n_j)) + 2

            # Calculate expected counts for the CSZIP model
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

            # Combine intervals where the expected counts are too small
            while f_exp[-1] < 0.25:
                Prob_Matrix[:, -2] = Prob_Matrix[:, -2] + Prob_Matrix[:, -1]
                Prob_Matrix = Prob_Matrix[:, 0:-1]
                Prob_mean[-2] = Prob_mean[-2] + Prob_mean[-1]
                Prob_mean = Prob_mean[0:-1]
                f_exp[-2] = f_exp[-2] + f_exp[-1]
                f_exp = f_exp[0:-1]

            # Calculate observed counts
            num_class = len(f_exp)
            f_obs = np.bincount(n_j.astype(np.int16), minlength=num_class)
            f_obs[num_class - 1] = np.sum(f_obs[num_class - 1:])
            f_obs = f_obs[:num_class]

            f_exp_i.append(f_exp)
            f_obs_i.append(f_obs)

            # When the number of groupings is less than or equal to 3, the data will be fitted exactly because the
            # degrees of freedom are 0, and thus it cannot be determined whether it fits the CSZIP model.
            if num_class <= 3:
                skip_number = skip_number + 1
                flag[i, j] = 2
                continue

            # Cell-specific chi-square test for CSZIP model
            D = len(n_j) * np.diag(Prob_mean)
            for k in range(len(n_j)):
                exp_prob = Prob_Matrix[k, :]
                temp = exp_prob.reshape(-1, 1)
                D = D - temp * np.transpose(temp)

            f_exp_star = f_exp[0:-1]
            f_obs_star = f_obs[0:-1]
            D_star = D[0:-1, 0:-1]
            f_diff = f_obs_star - f_exp_star
            # f_diff = np.abs(f_obs_star - f_exp_star)-0.5 # Yates Continuity Correction
            f_diff = f_diff.reshape(-1, 1)
            chi_sq = np.dot(np.transpose(f_diff), np.linalg.solve(D_star, f_diff))
            chi_sq_dis = stats.chi2(df=len(f_exp) - 1 - 2)
            p_value[i, j] = 1 - chi_sq_dis.cdf(chi_sq)
            if p_value[i, j] > significance:
                flag[i, j] = 1

        f_obs_all.append(f_obs_i)
        f_exp_all.append(f_exp_i)

    return flag, p_value, f_obs_all, f_exp_all


def plot_new_RNA_counts_fitting_results(
        f_obs_CSP: DataFrame,
        f_obs_CSZIP: DataFrame,
        f_exp_CSP: DataFrame,
        f_exp_CSZIP: DataFrame,
        p_value_CSP: DataFrame,
        p_value_CSZIP: DataFrame,
        gene_name: list,
        figsize: tuple = (3, 3),
        dpi: int = 75,
):
    """ Plot the results of the comparison of observed counts, expected counts from the CSP model and expected counts
    from the CSZIP model for the given genes and each labeling duration.

    Args:
        f_obs_CSP: A DataFrame that holds the observed new mRNA counts for each gene and each labeling duration in the
        CSP model.
        f_obs_CSZIP: A DataFrame that holds the observed new mRNA counts for each gene and each labeling duration in the
        CSZIP model.
        f_exp_CSP: A DataFrame that holds the expected counts for the CSP model for each gene and each labeling
        duration.
        f_exp_CSZIP: A DataFrame that holds the expected counts for the CSZIP model for each gene and each labeling
        duration.
        p_value_CSP: A DataFrame that holds the P-value for hypothesis testing using the CSP model for each gene and
        each labeling duration.
        p_value_CSZIP: A DataFrame that holds the P-value for hypothesis testing using the CSZIP model for each gene and
        each labeling duration.
        gene_name: A list of gene names that are going to be visualized.
        figsize: The width and height of each panel in the figure.
        dpi: The dot per inch of the figure.

    Returns:
    -------
        A matplotlib plot that shows the results of the comparison of observed counts, expected counts from the CSP
        model and expected counts from the CSZIP model for the given genes and each labeling duration.

    """

    def _plot_bar_new_counts(labels, ob_counts, csp_counts, cszip_counts, number, min_len, gene_name, time, legend_flag,
                             p_CSP, p_CSZIP, figsize, dpi):
        """"Use the bar chart to plot the observed counts and the line chart to plot the CSP and CSZIP model fits."""

        width = 0.5
        ind = np.linspace(0, min_len, min_len)
        # Make a square figure
        fig = plt.figure(number, figsize=figsize, dpi=dpi)
        # fig = plt.figure(number)
        ax = fig.add_subplot(111)
        # Plot
        ax.bar(ind, ob_counts, width, color='c', label='Observed')
        ax.plot(ind, csp_counts, color='b', marker='o', markersize=1.5, linewidth=0.8, label='CSP expected')
        ax.plot(ind, cszip_counts, color='red', marker='x', markersize=1.5, linewidth=0.8, label='CSZIP expected')

        # Set the ticks on x-axis
        ax.set_xticks(ind)
        if len(labels) < 10:
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels(labels, rotation=45)

        # labels
        ax.set_xlabel('New Counts')
        # title
        ax.set_title(f"gene name:{gene_name} time:{time}")
        # Legend and p_value
        if legend_flag:
            plt.legend(fontsize=6)
            plt.text(0.74, 0.26, '$P_{CSP}=$' + ("%3.2f" % p_CSP) + '\n' + '$P_{CSZIP}=$' + ("%.2f" % p_CSZIP),
                     ha='center',
                     va='center', transform=ax.transAxes)
        else:
            plt.text(0.74, 0.76, '$P_{CSP}=$' + ("%3.2f" % p_CSP) + '\n' + '$P_{CSZIP}=$' + ("%.2f" % p_CSZIP),
                     ha='center',
                     va='center', transform=ax.transAxes)

        plt.grid(False)
        plt.show()
        # plt.savefig(f'{save_name}gene_{gene_name}_time{time: .2f}.pdf', dpi=dpi, ftransparent=True,
        #             bbox_inches="tight")


    time = f_obs_CSP.columns.tolist()
    for j in range(len(time)):
        # Get the data used for drawing
        f_obs_csp = f_obs_CSP[time[j]][gene_name]
        f_exp_csp = f_exp_CSP[time[j]][gene_name]
        f_obs_cszip = f_obs_CSZIP[time[j]][gene_name]
        f_exp_cszip = f_exp_CSZIP[time[j]][gene_name]
        p_value_csp = p_value_CSP[time[j]][gene_name]
        p_value_cszip = p_value_CSZIP[time[j]][gene_name]

        # Unify the interval division of CSP and CSZIP models
        len_csp = len(f_obs_csp)
        len_cszip = len(f_obs_cszip)
        min_len = min(len_cszip, len_csp)
        if len_csp > len_cszip:
            f_obs_csp[len_cszip - 1] = np.sum(f_obs_csp[len_cszip - 1:])
            f_exp_csp[len_cszip - 1] = np.sum(f_exp_csp[len_cszip - 1:])
            f_obs_csp = f_obs_csp[0:len_cszip]
            f_exp_csp = f_exp_csp[0:len_cszip]
        elif len_cszip > len_csp:
            f_obs_cszip[len_csp - 1] = np.sum(f_obs_cszip[len_csp - 1:])
            f_exp_cszip[len_csp - 1] = np.sum(f_exp_cszip[len_csp - 1:])
            f_obs_cszip = f_obs_cszip[0:len_csp]
            f_exp_cszip = f_exp_cszip[0:len_csp]

        # Convert counts to frequency
        f_obs_csp = f_obs_csp / np.sum(f_obs_csp)
        f_exp_csp = f_exp_csp / np.sum(f_exp_csp)
        f_exp_cszip = f_exp_cszip / np.sum(f_exp_cszip)

        x = np.arange(min_len)
        x_labels = x.astype(str)
        x_labels[-1] = '>' + str(min_len - 2)
        # For better visualization, only draw the legend on the figure with the first labeling duration.
        if j == 0:
            _plot_bar_new_counts(x_labels, f_obs_csp, f_exp_csp, f_exp_cszip, j, min_len, gene_name, time[j],
                                 True, p_value_csp, p_value_cszip, figsize, dpi)
        else:
            _plot_bar_new_counts(x_labels, f_obs_csp, f_exp_csp, f_exp_cszip, j, min_len, gene_name, time[j],
                                 False, p_value_csp, p_value_cszip, figsize, dpi)


def plot_total_RNA_counts_fitting_results(
        f_obs: Series,
        p_value: Series,
        gene_name_list: list,
        figsize: tuple = (3, 3),
        dpi: int = 75,
        save_path: Optional[str] = None
):
    """ Plot the comparison of total mRNA counts between different labeling durations

    Args:
        f_obs: A Series that holds the observed total mRNA counts for each gene and each labeling duration.
        p_value: A Series that holds the P-value of the chi-square independence test.
        gene_name_list: A list of gene names that are going to be visualized.
        figsize: The width and height of each panel in the figure.
        dpi: The dot per inch of the figure.
        save_path: The save path for visualization results. save_name = None means that only show but not save the
        results.

    Returns:
        -------
        A matplotlib plot that shows the comparison of total mRNA counts between different labeling durations
    """

    def plot_bar_total_counts(ob_counts, number, gene_name, time, legend_flag, p_value, figsize, dpi, save_path):
        """ Plot the comparison of total mRNA counts between different labeling durations using bar charts"""
        groups = len(ob_counts)
        Y = np.zeros(shape=(groups, len(time)))

        # Convert counts to frequency
        for j in range(len(time)):
            Y[:, j] = ob_counts.values[:, j]
            Y[:, j] = Y[:, j] / np.sum(Y[:, j])
        Y_max = np.max(Y, axis=1)

        # For better visualization, intervals with frequencies less than 0.01 were merged.
        index = np.where(Y_max < 0.01)
        temp_index = int(index[0][0])
        Y[temp_index, :] = np.sum(Y[temp_index:, :], axis=0)
        Y = Y[0:temp_index + 1, :]

        groups, _ = np.shape(Y)
        fig = plt.figure(number + 1, figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        x_labels = ob_counts.index.tolist()
        del x_labels[temp_index + 1:]
        x_labels[-1] = '>' + str(int(x_labels[-1]) - 1)
        width = 0.15
        ind = np.linspace(0, groups, groups)

        # Set the ticks on x-axis
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels)
        # labels
        ax.set_xlabel('Total Counts')
        ax.set_ylabel('Total RNA Frequency')

        for j in range(len(time)):
            ax.bar(ind - 2.5 * width + j * width, Y[:, j], width, label=("%d mins" % (time[j] * 60)))

        if legend_flag:
            plt.legend()
            plt.text(0.75, 0.15, '$P=$' + ("%0.2f" % p_value), ha='center', va='center', transform=ax.transAxes)
        else:
            plt.text(0.75, 0.75, '$P=$' + ("%0.2f" % p_value), ha='center', va='center', transform=ax.transAxes)

        plt.grid(False)
        plt.title(gene_name)
        if save_path:
            plt.savefig(f'{save_path}gene_{gene_name}.pdf', dpi=dpi, bbox_inches="tight")
        plt.show()
        # plt.close()

    for i in range(len(gene_name_list)):
        # Get the data used for drawing
        gene_name = gene_name_list[i]
        cur_f_obs = f_obs[gene_name]
        cur_p_value = p_value[gene_name]
        time = cur_f_obs.columns.tolist()
        # For better visualization, only draw the legend on the figure of the first gene.
        if i == 0:
            plot_bar_total_counts(cur_f_obs, i, gene_name, time, True, cur_p_value, figsize, dpi, save_path)
        else:
            plot_bar_total_counts(cur_f_obs, i, gene_name, time, False, cur_p_value, figsize, dpi, save_path)
