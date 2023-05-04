import numpy as np
import pandas as pd
import scipy

# Structures for returning results

class MANOVAResult:
    def __init__(self, beta_hat, E, B, H, wilks_lambda, pillais_trace, hotelling_lawley_trace, roys_largest_root):
        self.beta_hat = beta_hat
        self.E = E
        self.B = B
        self.H = H
        self.wilks_lambda = wilks_lambda
        self.pillais_trace = pillais_trace
        self.hotelling_lawley_trace = hotelling_lawley_trace
        self.roys_largest_root = roys_largest_root


class F_statistic:
    def __init__(self, statistic, F, df_n, df_d, p_value):
        self.statistic = statistic
        self.F = F
        self.df_n = df_n
        self.df_d = df_d
        self.p_value = p_value


class chi2_statistic:
    def __init__(self, statistic, chi2, df, p_value):
        self.statistic = statistic
        self.chi2 = chi2
        self.df = df
        self.p_value = p_value


# Data processing functions

def create_design_matrix(df, column):
    """
    The model is a ``reduced-rank factor effects'' model in the
    style of SAS. The X matrix is the ``model'' or ``design''
    matrix. It leads with a column of ones for the intercept,
    after which it has an indicator column for each observed
    variable.
    """
    X = np.hstack([
        np.ones((df.shape[0], 1)),
        pd.get_dummies(df[column]).values
    ])
    return(X)


def create_response_matrix(data, columns):
    """
    The response matrix should just be the observations, all the
    observations, and only the observations. Group indicators
    should be omitted and are instead represented by the model
    matrix above.

    Yes, this code is trivial; it is designed to make clear what
    is expected / required.
    """
    Y = data[columns].values
    return(Y)


# Helper tools to create contrast matrices

def create_contrast_type_iii(X):
    """
    Creates a ``Type III'' hypothesis matrix intended to test
    whether any of the means of any of the groups differ. Note
    that this assumes that such a test is sensible, which
    depends very much on the data and what questions you're
    trying to answer.
    """
    n, r = X.shape
    C = np.zeros((r - 2, r))
    for i in range(1, r - 1):
        C[i - 1, i] = -1
        C[i - 1, i + 1] = 1
    return C


def orthopolynomial_contrasts(n, degree):
    """
    n: number of contrasts (predictor variables)
    degree: highest polynomial degree to include
    """

    x = np.linspace(-1, 1, n)
    M = np.empty((degree, n))
    M[0, :] = x
    if degree > 1:
        M[1, :] = (3 * x ** 2 - 1) / 2
    for i in range(2, degree):
        M[i, :] = ((2 * i + 1) * x * M[i - 1, :] - i * M[i - 2, :]) / (i + 1)
    return M.T


# Statistical tests

def run_manova(X, Y, C, M, alpha=0.05):
    """
    X: model (column of ones followed by ``dummies'' for groups)
    Y: data (rows must match X)
    C: contrast across variables
    M: contrast across groups

    Calculations here follow the math given in:
    https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/statug/statug_introreg_sect038.htm

    We mostly follow the variable naming convention in SAS;
    however, where we use `C` for the contrast matrix across
    variables, the SAS documentation uses `L`. There may be other
    differences.

    We test the null hypothesis `C @ beta @ M = 0.

    That is, with whatever contrasts you supply in the C and M
    matrices, we test the assumption that there is no
    significant variation.

    If the resulting p-value is less than your choice of alpha,
    you should reject the null hypothesis and conclude that
    there is variation. But you may need to resort to other
    tests to determine what is varying. That might include
    different choices of contrasts, univariate tests, or other
    techniques entirely.
    """

    beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ Y

    # E, the error / within-group SSCP matrix (AKA ``W'')
    E = M.T @ (
        Y.T @ Y
        -
        Y.T @ X @ np.linalg.pinv(X.T @ X) @ X.T @ Y
    ) @ M

    # B, the between-group SSCP matrix
    B =   (C @ beta_hat).T \
        @ np.linalg.inv(C @ np.linalg.pinv(X.T @ X) @ C.T) \
        @ C @ beta_hat

    # H, the hypothesis SSCP matrix
    H = M.T @ B @ M

    n, p = Y.shape
    g = X.shape[1] - 1
    q = np.linalg.matrix_rank(C @ np.linalg.pinv(X.T @ X) @ C.T)

    wl = wilks_lambda(E, H, n, p, g, q)
    pt = pillais_trace(E, H, n, p, g, q)
    hlt = hotelling_lawley_trace(E, H, n, p, g, q)
    rlr = roys_largest_root(E, H, n, p, g)

    return MANOVAResult(beta_hat, E, B, H, wl, pt, hlt, rlr)


def perform_box_m_test(X, Y):
    """
    Compute Box's M test for the homogeneity of covariance matrices.

    Parameters:
    X (numpy array): A 2D numpy array representing the model matrix (including a leading column of ones and columns of dummy variables for group inclusion).
    Y (numpy array): A 2D numpy array representing the observations.

    Returns a chi2_statistic object.
    """
    num_groups = X.shape[1] - 1
    num_variables = Y.shape[1]
    num_observations = [np.sum(X[:, i+1]) for i in range(num_groups)]

    groups = [Y[X[:, i+1].astype(bool)] for i in range(num_groups)]
    means = [np.atleast_2d(np.mean(group, axis=0)).T for group in groups]
    covariances = [np.cov(group, rowvar=False) for group in groups]

    pooled_covariance = calculate_pooled_covariance_matrix(X, Y)

    u = 0
    M = 0

    for n_i, cov_i in zip(num_observations, covariances):
        u += 1 / (n_i - 1)
        M += (n_i - 1) * np.log(np.linalg.det(cov_i))

    u = (u - (1 / (sum(num_observations) - num_groups))) * (
            (2 * num_variables**2 + 3 * num_variables - 1)
            /
            (6 * (num_variables + 1) * (num_groups - 1))
        )

    M = (sum(num_observations) - num_groups) * np.log(np.linalg.det(pooled_covariance)) - M

    C = (1 - u) * M

    nu = 0.5 * num_variables * (num_variables + 1) * (num_groups - 1)

    p_value = 1 - scipy.stats.chi2.cdf(C, nu)

    return chi2_statistic(M, C, nu, p_value)


def mauchly(X, Y):
    """
    X: model (column of ones followed by ``dummies'' for groups)
    Y: data (rows must match X)
    """

    n = Y.shape[1]
    degree = n - 1
    M = orthopolynomial_contrasts(n, degree)
    S_p = calculate_pooled_covariance_matrix(X, Y)

    k = M.shape[1]
    lsl_matrix = M.T @ S_p @ M
    determinant = np.linalg.det(lsl_matrix)
    trace = np.trace(lsl_matrix)
    w_stat = determinant / ((1 / (k - 1)) * trace) ** (k - 1)

    # Calculate the transformed W statistic that is chi-square distributed
    n, _ = X.shape
    n1 = n - 1
    g = 1 - (2 * k ** 2 + k + 2) / (6 * k * n1)
    transformed_w_stat = -n1 * g * np.log(w_stat)
    df = k * (k + 1) / 2 - 1

    p_value = 1 - scipy.stats.chi2.cdf(transformed_w_stat, df)
    return(chi2_statistic(w_stat, transformed_w_stat, df, p_value))


# Multivariate test statistics

def wilks_lambda(E, H, n, p, g, q):
    """
    n: number of observations (rows)
    p: number of variables (columns in Y)
    g: number of groups (columns in X excluding the column of leading ones)
    q: the rank of (C @ np.linalg.pinv(X.T @ X) @ C.T)

    The calculation is ``exact'' in the case of g = 3 and n >= 1;
    otherwise, it's an approximation.
    """
    wilks_lambda = np.linalg.det(E) / np.linalg.det(H + E)
    if wilks_lambda < 1e-15:
        wilks_lambda = 1e-15

    # ``Exact'' calculation in limited circumstances: three groups
    # and at least one dependent variable. Recall that X leads
    # with a column of ones, so a three-group analysis will have
    # four columns in X.
    if g == 3 and n >= 1:

        F = ((n - p - 2) / p) * ((1 - np.sqrt(wilks_lambda)) / np.sqrt(wilks_lambda))
        df_n = 2 * p
        df_d = 2 * (n - p - 2)

    else:
        v = n*(n+1)/2
        p = np.linalg.matrix_rank(H+E)
        s = min(p, q)
        m = (abs(p-q)-1)/2
        n = (v-p-1)/2
        r = v - (p-q+1)/2
        u = (p*q-2)/4
        if p**2 + q**2 - 5 > 0:
            t = np.sqrt( (p^2 * q^2 - 4) / (p^2 + q^2 - 5) )
        else:
            t = 1

        F = ( (1 - wilks_lambda**(1/t)) / wilks_lambda**(1/t) ) * ( (r*t - 2*u) / p*q )
        df_n = p*2
        df_d = r - 2*u

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(wilks_lambda, F, df_n, df_d, p_value)


def pillais_trace(E, H, n, p, g, q):
    """
    n: number of observations (rows)
    p: number of variables (columns in Y)
    g: number of groups (columns in X excluding the column of leading ones)
    """

    V = np.trace(H @ np.linalg.inv(H+E) )

    s = g - 1

    err_dof = n - q - 1
    p = np.linalg.matrix_rank(H+E)
    s = min(p, q)
    m = ( np.abs(p - q) - 1) / 2
    n = (err_dof - p - 1) / 2

    df_n = s * (2*m + s + 1)
    df_d = s * (2*n + s + 1)

    F = ( (2*n + s + 1) / (2*m + s + 1) ) * (V / (s-V))

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(V, F, df_n, df_d, p_value)


def hotelling_lawley_trace(E, H, n, p, g, q):
    """
    n: number of observations (rows)
    p: number of variables (columns in Y)
    g: number of groups (columns in X excluding the column of leading ones)
    """

    U = np.trace(np.linalg.inv(E) @ H)

    s = g - 1

    err_dof = n - q - 1
    p = np.linalg.matrix_rank(H+E)
    s = min(p, q)
    m = ( np.abs(p - q) - 1) / 2
    n = (err_dof - p - 1) / 2

    df_n = s * (2*m + 2 + 1)
    # NOTE: The following calculation is what is specified in the
    # SAS documentation. The result does not match calculations
    # done with SAS. Also note that the calculated F statistic
    # differs accordingly as well.
    df_d = 2 * (s*n + 1)

    F = ( (2 * (s*n + 1)) * U) / (s**2 * (2*m + s + 1) )

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(U, F, df_n, df_d, p_value)


def roys_largest_root(E, H, n, p, g):
    """
    n: number of observations (rows)
    p: number of variables (columns in Y)
    g: number of groups (columns in X excluding the column of leading ones)
    """
    largest_root = np.max(np.real(np.linalg.eigvals(np.linalg.inv(E) @ H)))

    s = g - 1

    df_n = p
    df_d = n - p - s + 1

    F = largest_root * (n - p - s + 1) / p

    p_value = scipy.stats.f.sf(largest_root, df_n, df_d)

    return F_statistic(largest_root, F, df_n, df_d, p_value)


# Post-hoc tests

def greenhouse_geisser_correction(Y, M):
    S = np.cov(Y - Y.mean(axis=0), rowvar=False)
    Sigma_m = M.T @ S @ M
    eigenvalues = np.linalg.eigvalsh(Sigma_m)
    epsilon = \
        (np.sum(eigenvalues) ** 2) \
        / \
        ((M.shape[0] - 1) * np.sum(eigenvalues ** 2))
    return epsilon

def tukey_test():
    pass

def bonferroni_correction():
    pass


# Utility functions

def calculate_pooled_covariance_matrix(X, Y):
    n, r = X.shape
    p = Y.shape[1]
    S_p = np.zeros((p, p))

    group_ids = np.unique(X[:, 1:])

    for group_id in group_ids:
        group_idx = (X[:, 1:] == group_id).any(axis=1)
        Y_group = Y[group_idx, :]
        n_group = Y_group.shape[0]
        group_cov = np.cov(Y_group, rowvar=False, ddof=1)
        S_p += (n_group - 1) * group_cov

    S_p /= (n - r)

    return S_p
