import numpy as np
import pandas as pd
import scipy

# Structures for returning results

class MANOVAResult:
    def __init__(self, b, E, H, wilks_lambda, pillais_trace, hotelling_lawley_trace, roys_largest_root):
        self.b = b
        self.E = E
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
    The model is a ``rank-deficient factor effects'' model in the
    style of SAS.  The X matrix is the ``model'' or ``design''
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

def create_type_iii_hypothesis_contrast(X):
    """
    Creates a ``Type III'' hypothesis matrix intended to test
    whether any of the means of any of the groups differ. Note
    that this assumes that such a test is sensible, which
    depends very much on the data and what questions you're
    trying to answer.

    See:

    https://documentation.sas.com/doc/en/statug/15.2/statug_introglmest_sect015.htm

    and

    https://documentation.sas.com/doc/en/statug/15.2/statug_introglmest_toc.htm
    """
    n, r = X.shape
    L = np.zeros((r - 2, r))
    for i in range(1, r - 1):
        L[i - 1, i] = -1
        L[i - 1, i + 1] = 1
    return L


def create_orthopolynomial_contrasts(levels, degree):
    """
    levels: an one-dimensional NumPy array specifying the
    treatment levels. For p equally-spaced time intervals, use
    np.array([1, 2, ..., p]). For different doses, this should be
    a list of the doses. There must be as many elements in levels
    as there are variables in Y or else you will get a matrix
    dimension mismatch error.

    degree: the highest-degree polynomial to create. For Mauchly's
    test, this should be one less than the number of levels.

    See:

    https://documentation.sas.com/doc/en/statug/15.2/statug_glm_details46.htm
    """

    Y = levels - np.mean(levels)
    X = np.zeros([len(levels), degree + 1])
    for i in range(degree + 1):
        X[:,i] = Y**i
    Q, R = np.linalg.qr(X)
    return(np.delete(Q, 0, 1))


# Statistical tests

def run_manova(X, Y, L, M, alpha=0.05):
    """
    X: model (column of ones followed by ``dummies'' for groups)
    Y: data (rows must match X)
    L: contrast across variables
    M: contrast across groups

    Calculations here follow the math given in:

    https://documentation.sas.com/doc/en/statug/15.2/statug_glm_details45.htm

    and

    https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm

    We mostly follow the variable naming convention in SAS;
    however, SAS itself isn't consisten. For example, they use
    both `b` (as we do here) and `B` for ``beta hat,'' the
    esimation of the parameter matrix.

    We test the null hypothesis `L @ beta @ M = 0`.
    """

    # Estimator for beta, the matrix of parameter esimates:
    b = np.linalg.pinv(X.T @ X) @ X.T @ Y
    # (Note: SAS's documentation is inconsistent with its
    # designation of this matrix, using both upper- and lower-case
    # ``B'' or ``b'' for this matrix.

    # Hypothesis SSCP matrix H, associated with the hypothesized effect:
    H =   M.T \
        @ (L @ b).T \
        @ np.linalg.inv(
              L \
            @ np.linalg.pinv(X.T @ X) \
            @ L.T
        ) \
        @ (L @ b) \
        @ M
        
    # Error SSCP matrix E, associated with the error effect:
    E =  M.T @ (Y.T @ Y - b.T @ (X.T @ X) @ b) @ M

    # Note: the diagonal elements of H and E correspond to the
    # hypothesis and error SS for univariate tests.

    p = np.linalg.matrix_rank(H+E)
    q = np.linalg.matrix_rank(L @ np.linalg.pinv(X.T @ X) @ L.T)
    v = X.shape[0] - np.linalg.matrix_rank(X) # Error degrees of freedom
    s = min(p, q)
    m = (np.abs(p-q)-1)/2
    n = (v-p-1)/2

    wl = wilks_lambda(E, H, p, q, v, s, m, n)
    pt = pillais_trace(E, H, p, q, v, s, m, n)
    hlt = hotelling_lawley_trace(E, H, p, q, v, s, m, n)
    rlr = roys_largest_root(E, H, p, q, v, s, m, n)

    return MANOVAResult(b, E, H, wl, pt, hlt, rlr)


def perform_box_m_test(X, Y):
    """
    Compute Box's M test for the homogeneity of covariance matrices.

    Parameters:
    X (numpy array): A 2D numpy array representing the model matrix (including a leading column of ones and columns of dummy variables for group inclusion).
    Y (numpy array): A 2D numpy array representing the observations.

    Returns a chi2_statistic object.

    See Johnson & Wichern page 310.
    """

    p = Y.shape[1] # number of variables
    g = X.shape[1] - 1 # number of groups
    n_l = [np.sum(X[:, i+1]) for i in range(g)] # sample size for the lth group

    groups = [Y[X[:, i+1].astype(bool)] for i in range(g)]
    means = [np.mean(group, axis=0).reshape(-1,1) for group in groups]
    covariances = [np.cov(group, rowvar=False) for group in groups]

    S_p = calculate_pooled_covariance_matrix(X, Y)

    temp_1 = 0
    temp_2 = 0
    for n_i, cov_i in zip(n_l, covariances):
        temp_1 += (n_i - 1) * np.log(np.linalg.det(S_p))
        temp_2 += (n_i - 1) * np.log(np.linalg.det(cov_i))
    M = temp_1 - temp_2

    temp_1 = 0
    temp_2 = 0
    for n_i in n_l:
        temp_1 += 1 / (n_i - 1)
        temp_2 += n_i - 1

    u = (temp_1 - (1/temp_2)) * (
            (2 * p**2 + 3 * p - 1)
            /
            (6 * (p + 1) * (g - 1))
        )

    C = (1 - u) * M

    nu = 0.5 * p * (p + 1) * (g - 1)

    p_value = 1 - scipy.stats.chi2.cdf(C, nu)

    return chi2_statistic(M, C, nu, p_value)


def mauchly(X, Y):
    """
    X: model (column of ones followed by ``dummies'' for groups)
    Y: data (rows must match X)

    Mauchly's test of sphericity is for a repeated measures
    analysis, which requires an M matrix of certain dimensions. We
    calculate here the simplest type, of equally-spaced
    polynomials. If you are testing something different, you will
    need to modify this code. I considered adding a parameter to
    pass the levels ... but that requires more error handling than
    I have time for at the moment.

    See STP 533 class note from Dr. Mark Reiser
    """

    p = Y.shape[1]
    k = p - 1
    M = create_orthopolynomial_contrasts(np.array(np.arange(p))+1, k)
    S_p = calculate_pooled_covariance_matrix(X, Y)

    W = np.linalg.det(M.T @ S_p @ M) / ( k**-1 * np.trace(M.T @ S_p @ M) )**k

    n_1 = X.shape[0] - np.linalg.matrix_rank(X)
    d = 1 - (2 * k**2 + k + 2) / (6 * k * n_1)

    chi2 = -n_1 * d * np.log(W)

    df = ( (k * (k+1)) / 2 ) - 1

    p_value = 1 - scipy.stats.chi2.cdf(chi2, df)
    return(chi2_statistic(W, chi2, df, p_value))


# Multivariate test statistics

def wilks_lambda(E, H, p, q, v, s, m, n):

    wilks_lambda = np.linalg.det(E) / np.linalg.det(H + E)
    if wilks_lambda < 1e-15:
        wilks_lambda = 1e-15

    r = v - ( (p-q+1) / 2 )
    u = (p*q-2)/4
    if p**2 + q**2 - 5 > 0:
        t = np.sqrt( (p**2 * q**2 - 4) / (p**2 + q**2 - 5) )
    else:
        t = 1

    F =   ( (1 - wilks_lambda**(1/t) ) / wilks_lambda**(1/t) ) \
        * ( (r*t - 2*u) / (p*q) )

    df_n = p*q
    df_d = r*t - 2*u

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(wilks_lambda, F, df_n, df_d, p_value)


def pillais_trace(E, H, p, q, v, s, m, n):

    V = np.trace(H @ np.linalg.inv(H+E) )

    F = ( (2*n + s + 1) / (2*m + s + 1) ) * (V / (s-V))

    df_n = s * (2*m + s + 1)
    df_d = s * (2*n + s + 1)

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(V, F, df_n, df_d, p_value)


def hotelling_lawley_trace(E, H, p, q, v, s, m, n):

    U = np.trace(np.linalg.inv(E) @ H)

    F = ( (2 * (s*n + 1)) * U) / (s**2 * (2*m + s + 1) )

    df_n = s * (2*m + 2 + 1)
    # NOTE: The following calculation is what is specified in the
    # SAS documentation. The result does not match calculations
    # done with SAS. Also note that the calculated F statistic
    # differs accordingly as well.
    #
    # Further note: both `s` and `n` are integers, so the result
    # of this computation must also be an integer. But SAS
    # regularly reports a fractional value.
    df_d = 2 * (s*n + 1)

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(U, F, df_n, df_d, p_value)


def roys_largest_root(E, H, p, q, v, s, m, n):

    Theta = np.max(np.real(np.linalg.eigvals(np.linalg.inv(E) @ H)))

    # Note: r is `max(p, q)`; s is `min(p, q)`.
    r = max(p, q)

    F = Theta * (v - r + q) / r

    df_n = r
    df_d = v - r + q

    p_value = scipy.stats.f.sf(F, df_n, df_d)

    return F_statistic(Theta, F, df_n, df_d, p_value)


# Post-hoc tests

def greenhouse_geisser_correction(X, Y, M, calculation_method="Eigenvalues"):
    """
    The parameters should be prepared the same as for
    `perform_manova` above.
    """

    S_p = calculate_pooled_covariance_matrix(X, Y)

    # Note that both calculations match to within 14 decimal places or so.

    if calculation_method == "Eigenvalues":
        # "Modern" calculation with eigenvalues
        # https://real-statistics.com/anova-repeated-measures/sphericity/additional-information-sphericity/

        Sigma_m = M.T @ S_p @ M
        eigenvalues = np.linalg.eigvalsh(Sigma_m)
        epsilon = \
            (np.sum(eigenvalues) ** 2) \
            / \
            ((M.shape[0] - 1) * np.sum(eigenvalues ** 2))

    else:
        # Original calculation in PSYCOMETRIKA~VOL.24, NO. 2 JUNE, 1959
        # https://www.ece.uvic.ca/~bctill/papers/mocap/Greenhouse_Geisser_1959.pdf

        p = Y.shape[1]

        temp_1 = 0
        for i in range(p):
            for j in range(p):
                temp_1 += S_p[i,j]**2

        temp_2 = 0
        for i in range(p):
            temp_2 += np.mean(S_p[:,i])**2

        epsilon = ( p**2 * (np.mean(np.diagonal(S_p)) - np.mean(S_p))**2 ) \
                / ( (p-1) * (temp_1 - 2*p*temp_2 + p**2 * np.mean(S_p)**2) )

    return epsilon

def tukey_test():
    pass

def bonferroni_correction():
    pass


# Utility functions

def calculate_pooled_covariance_matrix(X, Y):
    """
    See Johnson & Wichern page 310
    """

    p = Y.shape[1] # number of variables
    g = X.shape[1] - 1 # number of groups
    n_l = [np.sum(X[:, i+1]) for i in range(g)] # sample size for the lth group

    groups = [Y[X[:, i+1].astype(bool)] for i in range(g)]
    means = [np.mean(group, axis=0).reshape(-1,1) for group in groups]
    covariances = [np.cov(group, rowvar=False) for group in groups]

    temp_1 = 0
    temp_2 = 0

    for n_i, cov_i in zip(n_l, covariances):
        temp_1 += (n_i - 1) * cov_i
        temp_2 += n_i - 1

    S_p = temp_1 / temp_2

    return S_p
