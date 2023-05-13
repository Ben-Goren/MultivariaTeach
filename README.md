# MultivariaTeach

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Testing](#testing)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Introduction

MultivariaTeach is designed to help students of multivariate statistics understand and perform a selection of fundamental tests. Where many statistical packages provide "helpful" features that automatically perform various functions or hide “useless” information, the goal of this package is to provide a selection of simple tools which can be easily put together as desired. For example, rather than use a command to indicate you wish to perform a repeated measures analysis analysis of variance (ANOVA), there is a tool to create a matrix of polynomial contrasts which you can then use in the rest of the analysis.

## Prerequisites

In addition to Python, you will need the standard `numpy`, `pandas`, and `scipy` packages installed (which is almost certainly already the case). Almost all of the mathematical calculations are done with `numpy`; the `pandas` library is used in the initial transformation of data into the necessary format; and `scipy` is used to calculate $p$-values. If you wish to follow along with some of the examples in this README, you will also need the `sklearn` package for its `datasets`.

It is assumed that you are a graduate-level student in a multivariate statistics course. The math won't make much sense to those without the necessary academic prerequisites; and, to those who have completed such a course, the more common statistical analysis software packages (such as SAS or R) are almost certainly more practical.

## Installation

To install the `multivariateach` package, simply run the following command:

```bash
pip install multivariateach
```

## Usage

### A Note About SAS Documentation

Before detailing how to use MultivariaTeach, it should be noted that the math it implements (described below) is heavily influenced by and modeled on the documentation provided online by SAS.

However, that documentation is not entirely consistent.

Below, you will find references to the SAS documentation, including where conflicts can be found and the particular implementation used by MultivariaTeach.

As much as possible, notation in this document and naming conventions in the code follow those provided by SAS in its documentation.

### MANOVA

#### Overview

For those familiar with SAS, MultivariTeach provides much (but certainly not all!) of the basic functionality of `PROC GLM`; that is, of testing the null hypothesis $H_0: \mathbf{L} \boldsymbol{\beta} \mathbf{M} = \mathbf{0}$, against the alternative $H_a: \mathbf{L} \boldsymbol{\beta}\mathbf{M} \neq 0$, where $\mathbf{L}$ is a matrix of contrasts across groups; $\boldsymbol{\beta}$ is the parameter matrix; and $\mathbf{M}$ is a matrix of contrasts across variables.

In most software packages, performing a MANOVA test is done by loading all the data into an object and then writing out a textual command mimicking mathematical syntax to indicate the test you wish to perform. For example, consider Ronald Fisher's famous 1936 iris flower data set, with the species in the first column named `group` and the four measurements in the remaining columns, labeled `x1`, `x2`, `x3`, and `x4`. In Python, you can easily load the data as follows:

```python
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
data = pd.DataFrame(np.hstack([iris.target.reshape(-1, 1), iris.data]))
data.columns = ['group', 'x1', 'x2', 'x3', 'x4']
```

Elsewhere, you might use something in the spirit of `manova data=data, test='group ~ x1 + x2 + x3 + x4'`. The software then "helpfully" does various transformations to the data and reports the various test statistics.

Here, instead, you must yourself prepare the $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{L}$, and $\mathbf{M}$ matrices which are used in the actual calcuations. Note that all software which performs a MANOVA does this, one way or another; it's just that you mostly don't ever get a chance to see how it happens. Further, there are different-but-equivalent mathematical approaches to take; here, we use the same rank-deficient "factor effects" model as SAS, but R uses the "full rank" model instead.

In our model, $\mathbf{L}$ and $\mathbf{M}$ are as described above: matrices of contrasts. Here, $\mathbf{X}$ is the "model matrix." It has a leading column of ones, followed by "dummy" indicators (either $1$ or $0$) that identify which group the observation (row) belongs to. Then $\mathbf{Y}$ is simply the observations themselves.

While you are welcome to construct all four matrices by hand, MultivariaTeach provides tools (described below) to help construct them.

#### Calculations

When you perform the MANOVA, the function first finds
$$
\mathbf{b} = (\mathbf{X}^\intercal \mathbf{X})^- \mathbf{X}^\intercal \mathbf{Y},
$$
where $\mathbf{A}^\intercal$ denotes the transpose of $\mathbf{A}$ and $\mathbf{A}^-$ denotes the Moore-Penrose generalized inverse of $\mathbf{A}$. This $\mathbf{b}$ is the matrix of parameter estimates, elsewhere often denoted as $\hat{\boldsymbol{\beta}}$. Note that the SAS documentation sometimes identifies this matrix as $\mathbf{B}$.

Then, the function calculates:
$$
\mathbf{H} = \mathbf{M}^\intercal (\mathbf{L}\mathbf{b})^\intercal(\mathbf{L}(\mathbf{X}^\intercal\mathbf{X})^-\mathbf{L}^\intercal)^{-1}(\mathbf{L}\mathbf{b})\mathbf{M},
$$
where $\mathbf{A}^{-1}$ denotes the "regular" (non-generalized) inverse of $\mathbf{A}$. The $\mathbf{H}$ matrix is the hypothesis SSCP matrix associated with the hypothesized effect.

Next, the error SSCP matrix $\mathbf{E}$ associated with the error effect is calculated:
$$
\mathbf{E} = \mathbf{H}^\intercal (\mathbf{Y}^\intercal \mathbf{Y} - \mathbf{b}^\intercal (\mathbf{X}^\intercal\mathbf{X}) \mathbf{b})\mathbf{M}
$$
The $\mathbf{E}$ and $\mathbf{H}$ matrices are then used to calculate the test statistics.

Note that these calculations match those in the SAS documentation of the [Multivariate Analysis of Variance for the GLM Procedure](https://documentation.sas.com/doc/en/statug/15.2/statug_glm_details45.htm). However, in the SAS documentation of [Multivariate Tests](https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm) in its Introduction to Regression Procedures, these matrices are given as a general case using weighted regression capable of testing a null hypothesis where the right-hand-side is non-zero. Of course, when the weights are equal and the right-hand-size is zero, both calculations are equivalent.

#### Walkthrough

First, as usual, make sure you load the package at the top of your Python file:

```python
import multivariateach as mt
```

This `import` statement would typically be included with the lines above importing `pandas` and `sklearn`. We can now create our $\mathbf{X}$ and $\mathbf{Y}$ matrices:

```python
X = mt.create_design_matrix(data, 'group')
Y = mt.create_response_matrix(data, ['x1', 'x2', 'x3', 'x4'])
```

You can inspect these two matrices, if you wish, before performing the analysis.

For this example, we conduct a "Type III" hypothesis test as described in the [Type III and IV SS and Estimable Functions](https://documentation.sas.com/doc/en/statug/15.2/statug_introglmest_sect015.htm), section of the SAS manual on [The Four Types of Estimable Functions](https://documentation.sas.com/doc/en/statug/15.2/statug_introglmest_toc.htm). This choice of hypothesis dictates our construction of $\mathbf{L}$. Note that this hypothesis will not make sense in many real-world analyses; however, it is the default analysis performed by SAS and is, indeed, suitable for many textbook (and real-world) examples.

Creating such a matrix with MultivariaTeach is trivial:

```python
L = mt.create_type_iii_hypothesis_contrast(X)
```

In many MANOVA tests, there is no need to examine transformations of variables, in which case one uses $\mathbf{M} = \mathbf{I}$. In this case, $H_0: \mathbf{L} \boldsymbol{\beta} \mathbf{M} = \mathbf{0} \equiv \mathbf{L} \boldsymbol{\beta} \mathbf{I} = 0 \equiv \mathbf{L} \boldsymbol{\beta} = 0$. To perform such a test in MultivariaTeach, use an identity matrix of the size equal to the number of variables in $\mathbf{Y}$ for your $\mathbf{M}$ matrix. You can do this with `numpy`, another library which you should import at the top of your file:

```python
import numpy as np

M = np.eye(Y.shape[1])
```

With all the building blocks assembled, you can now perform the MANOVA:

```python
results = mt.run_manova(X, Y, L, M)
```

Unlike with most other software, you are not presented with a pretty-formatted table of results; instead, the `results` object can now be itself queried for \ldots\ well, for anything and everything. So, for example, a simple-but-complete test might look like this:

```python
import numpy as np
import pandas as pd
from sklearn import datasets
import multivariateach as mt

iris = datasets.load_iris() # Fisher's 1936 flower data

data = pd.DataFrame(np.hstack([iris.target.reshape(-1, 1), iris.data])) # Extract the parts we want in the structure we need
data.columns = ['group', 'x1', 'x2', 'x3', 'x4'] # Give names to the columns

X = mt.create_design_matrix(data, 'group')
Y = mt.create_response_matrix(data, ['x1', 'x2', 'x3', 'x4'])
L = mt.create_type_iii_hypothesis_contrast(X)
M = np.eye(Y.shape[1])

results = mt.run_manova(X, Y, L, M)

print("L:\n", L)
print("M:\n", M)
print("b (aka beta hat):\n", results.b)
print("E (error SSCP):\n", results.E)
print("H (hypothesis SSCP):\n", results.H)
print(f"Wilks's Lambda: {results.wilks_lambda.statistic}; F value: {results.wilks_lambda.F}; Num DF: {results.wilks_lambda.df_n}; Den DF: {results.wilks_lambda.df_d}; Pr > F: {results.wilks_lambda.p_value}")
print(f"Pillai's Trace: {results.pillais_trace.statistic}; F value: {results.pillais_trace.F}; Num DF: {results.pillais_trace.df_n}; Den DF: {results.pillais_trace.df_d}; Pr > F: {results.pillais_trace.p_value}")
print(f"Hotelling-Lawley Trace: {results.hotelling_lawley_trace.statistic}; F value: {results.hotelling_lawley_trace.F}; Num DF: {results.hotelling_lawley_trace.df_n}; Den DF: {results.hotelling_lawley_trace.df_d}; Pr > F: {results.hotelling_lawley_trace.p_value}")
print(f"Roy's Largest Root: {results.roys_largest_root.statistic}; F value: {results.roys_largest_root.F}; Num DF: {results.roys_largest_root.df_n}; Den DF: {results.roys_largest_root.df_d}; Pr > F: {results.roys_largest_root.p_value}")
```

This will create the following output:

```[text]
L:
 [[ 0. -1.  1.  0.]
 [ 0.  0. -1.  1.]]
M:
 [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
b (aka beta hat):
 [[ 4.3825  2.293   2.8185  0.8995]
 [ 0.6235  1.135  -1.3565 -0.6535]
 [ 1.5535  0.477   1.4415  0.4265]
 [ 2.2055  0.681   2.7335  1.1265]]
E (error SSCP):
 [[38.9562 13.63   24.6246  5.645 ]
 [13.63   16.962   8.1208  4.8084]
 [24.6246  8.1208 27.2226  6.2718]
 [ 5.645   4.8084  6.2718  6.1566]]
H (hypothesis SSCP):
 [[ 63.2121 -19.9527 165.2484  71.2793]
 [-19.9527  11.3449 -57.2396 -22.9327]
 [165.2484 -57.2396 437.1028 186.774 ]
 [ 71.2793 -22.9327 186.774   80.4133]]
Wilks's Lambda: 0.023438630650878298; F value: 199.14534354008438; Num DF: 8; Den DF: 288.0; Pr > F: 1.3650058325896826e-112
Pillai's Trace: 1.1918988250414653; F value: 53.46648878461304; Num DF: 8.0; Den DF: 290.0; Pr > F: 9.74216271943989e-53
Hotelling-Lawley Trace: 32.47732024090117; F value: 580.5320993061084; Num DF: 8.0; Den DF: 286.0; Pr > F: 6.436176201236599e-172
Roy's Largest Root: 32.19192919827811; F value: 1166.9574334375816; Num DF: 4; Den DF: 145; Pr > F: 3.787297649634471e-109
```

With luck, by now the code above and its results are obvious and self-explanatory.

Also note that the $p$-values are given with their full precision \-\-\- an hundred and twelve leading zeros in the case of Wilks's Lambda above. As a general practice, when you report a $p$-value in your analysis, you should indicate it as being less than some arbitrarily small number, such as, "We find $p < 0.0001 < \alpha = 0.05$ and therefore reject $H_0$." If you use code to produce such output, be sure to check for vanishingly small $p$-values and adjust accordingly; MultivariaTeach will report the results of the calculation without adjustment.

#### Test statistics

The test statistics above are calculated as described in the SAS manual section on [Multivariate Tests](https://documentation.sas.com/doc/en/statug/15.2/statug_introreg_sect038.htm) mentioned above. Note again that this section documents the general case where one may test hypotheses where the right-hand-side is non-zero with weighted regression, neither of which is supported by MultivariaTeach.

Further especially note that the denominator degrees of freedom for the Hotelling-Lawley Trace in the manual are given by $2(sn+1)$, where $s$ and $n$ are calculated from matrix ranks and error degrees of freedom, all of which are integers. Yet SAS reports non-integer values for this value, including in the example on the page. In MultivariaTeach, the value is calculated as $2(sn+1)$, and so the degrees of freedom (and corresponding $F$ statistic and $p$-values) do not match those calculated by SAS.

##### Shared variables

The following variables are used by some or all of the calculations of the various statistics:

* $p$: the rank of $\mathbf{H}+\mathbf{E}$;
* $q$: the rank of $\mathbf{L}(\mathbf{X^\intercal \mathbf{X})^- \mathbf{L}^\intercal}$;
* $v$: the error degrees of freedom, calculated as the total observations less the rank of $\mathbf{X}$;
* $s$: the lesser of $p$ and $q$;
* $m = \frac{\lvert p-q \rvert - 1}{2}$; and
* $n = \frac{v-p-1}{2}$.

##### Wilks's Lambda

The statistic is calculated as:
$$
\lambda = \frac{ \lvert \mathbf{E} \rvert }{ \lvert \mathbf{E} + \mathbf{H} \rvert }.
$$
Then, let $r = v-\frac{p-q+1}{2}$ and $u=\frac{pq-2}{4}$. Next,
$$
t = \begin{cases} \sqrt{\frac{p^2q^2-4}{p^2+q^2-5}}, & p^2+q^2-5>0 \\ 1 & \operatorname{otherwise}. \end{cases}
$$
Now,
$$
F = \frac{1-\lambda^{\frac{1}{t}}}{\lambda^{\frac{1}{t}}}\cdot \frac{rt-2u}{pq}
$$
has $pq$ and $rt-2u$ degrees of freedom.

##### Pillai's Trace

If
$$
V = \operatorname{tr}\left(\mathbf{H}{(\mathbf{H}+\mathbf{E})^{-1}}\right),
$$
then
$$
F = \frac{2n+s+1}{2m+s+1} \cdot \frac{V}{s-V}
$$
has $s(2m+s+1)$ and $s(2n+s+1)$ degrees of freedom.

##### Hotelling-Lawley Trace

Let
$$
U = \operatorname{tr}(\mathbf{E}^{-1}\mathbf{H}),
$$
so that
$$
F = \frac{2(sn+1)U}{s^2(sm+s+1)}
$$
has $s(2m+2+1)$ and $2(sn+1)$ degrees of freedom.

As mentioned above, both $s$ and $n$ are integers and so the denominator degrees of freedom must also be an integer; however, SAS computes non-integer values for the denominator degrees of freedom.

##### Roy's Largest Root

Last, let $\Theta$ be the largest eigenvalue of $\mathbf{E}^{-1}\mathbf{H}$ and $r$ be the larger of $p$ and $q$. (Recall that $s$ is the lesser of the two.) Then
$$
F = \frac{\Theta(v-r+q)}{r}
$$
has $r$ and $v-r+q$ degrees of freedom.

### Repeated measures

One can use MANOVA with an $\mathbf{M}$ matrix of polynomial contrasts across repeated measurements to perform an analysis of repeated measures.

The SAS manual section on [Repeated Measures Analysis of Variance](https://documentation.sas.com/doc/en/statug/15.2/statug_glm_details46.htm) provides as an example five levels of a drug corresponding to 1, 2, 5, 10, and 20 milligrams administered to different groups; that same $\mathbf{M}$ matrix can be created in MultivariTeach with the following:

```python
levels = np.array([1, 2, 5, 10, 20])
degree = 5
M = mt.create_orthopolynomial_contrasts(levels, degree)
```

Note that the SAS documentation gives the transpose of $\mathbf{M}$, which can be printed with:

```python
print(M.T)
```

which outputs:

```
[[-0.425  -0.3606 -0.1674  0.1545  0.7984]
 [ 0.4349  0.2073 -0.3252 -0.7116  0.3946]
 [-0.4331  0.1366  0.7253 -0.5108  0.0821]
 [-0.4926  0.78   -0.3744  0.0936 -0.0066]]
```

(Also note that there are differences in some of the signs of the elements of the matrix; these differences are inconsequential and represent different orientations of equivalent orthonormal bases.)

The iris data, of course, is not a repeated measures experiment, so performing such a test with the data would not make sense; however, the following repeated measures example is from [Littell, Pendergast, and Natarajan (2000) ](https://pubmed.ncbi.nlm.nih.gov/10861779/) of an examination of the effects of two drugs (`a` and `c`) and a placebo (`p`), with "FEV1" measurements of respiratory ability taken over eight one-hour intervals after treatment. The first column is the patient ID, which is not of interest in this analysis. The second is of a baseline FEV1 measurement made before administering the drug; one could include it in an analysis, but we do not do so here. The next eight columns are the after-administration measurements; and the last is the drug.

We load the data and perform the analysis as above with the Iris data, but with `M` as polynomial contrasts rather than the identity matrix:

```python
import numpy as np
import pandas as pd
import multivariateach as mt

raw_data = [
    (201, 2.46, 2.68, 2.76, 2.50, 2.30, 2.14, 2.40, 2.33, 2.20, 'a'),
    (202, 3.50, 3.95, 3.65, 2.93, 2.53, 3.04, 3.37, 3.14, 2.62, 'a'),
    (203, 1.96, 2.28, 2.34, 2.29, 2.43, 2.06, 2.18, 2.28, 2.29, 'a'),
    (204, 3.44, 4.08, 3.87, 3.79, 3.30, 3.80, 3.24, 2.98, 2.91, 'a'),
    (205, 2.80, 4.09, 3.90, 3.54, 3.35, 3.15, 3.23, 3.46, 3.27, 'a'),
    (206, 2.36, 3.79, 3.97, 3.78, 3.69, 3.31, 2.83, 2.72, 3.00, 'a'),
    (207, 1.77, 3.82, 3.44, 3.46, 3.02, 2.98, 3.10, 2.79, 2.88, 'a'),
    (208, 2.64, 3.67, 3.47, 3.19, 2.19, 2.85, 2.68, 2.60, 2.73, 'a'),
    (209, 2.30, 4.12, 3.71, 3.57, 3.49, 3.64, 3.38, 2.28, 3.72, 'a'),
    (210, 2.27, 2.77, 2.77, 2.75, 2.75, 2.71, 2.75, 2.52, 2.60, 'a'),
    (211, 2.44, 3.77, 3.73, 3.67, 3.56, 3.59, 3.35, 3.32, 3.18, 'a'),
    (212, 2.04, 2.00, 1.91, 1.88, 2.09, 2.08, 1.98, 1.70, 1.40, 'a'),
    (214, 2.77, 3.36, 3.42, 3.28, 3.30, 3.31, 2.99, 3.01, 3.08, 'a'),
    (215, 2.96, 4.31, 4.02, 3.38, 3.31, 3.46, 3.49, 3.38, 3.35, 'a'),
    (216, 3.11, 3.88, 3.92, 3.71, 3.59, 3.57, 3.48, 3.42, 3.63, 'a'),
    (217, 1.47, 1.97, 1.90, 1.45, 1.45, 1.24, 1.24, 1.17, 1.27, 'a'),
    (218, 2.73, 2.91, 2.99, 2.87, 2.88, 2.84, 2.67, 2.69, 2.77, 'a'),
    (219, 3.25, 3.59, 3.54, 3.17, 2.92, 3.48, 3.05, 3.27, 2.96, 'a'),
    (220, 2.73, 2.88, 3.06, 2.75, 2.71, 2.83, 2.58, 2.68, 2.42, 'a'),
    (221, 3.30, 4.04, 3.94, 3.84, 3.99, 3.90, 3.89, 3.89, 2.98, 'a'),
    (222, 2.85, 3.38, 3.42, 3.28, 2.94, 2.96, 3.12, 2.98, 2.99, 'a'),
    (223, 2.72, 4.49, 4.35, 4.38, 4.36, 3.77, 4.23, 3.83, 3.89, 'a'),
    (224, 3.68, 4.17, 4.30, 4.16, 4.07, 3.87, 3.87, 3.85, 3.82, 'a'),
    (232, 2.49, 3.73, 3.51, 3.16, 3.26, 3.07, 2.77, 2.92, 3.00, 'a'),
    (201, 2.30, 3.41, 3.48, 3.41, 3.49, 3.33, 3.20, 3.07, 3.15, 'c'),
    (202, 2.91, 3.92, 4.02, 4.04, 3.64, 3.29, 3.10, 2.70, 2.69, 'c'),
    (203, 2.08, 2.52, 2.44, 2.27, 2.23, 2.01, 2.26, 2.34, 2.44, 'c'),
    (204, 3.02, 4.43, 4.30, 4.08, 4.01, 3.62, 3.23, 2.46, 2.97, 'c'),
    (205, 3.26, 4.55, 4.58, 4.44, 4.04, 4.33, 3.87, 3.75, 3.81, 'c'),
    (206, 2.29, 4.25, 4.37, 4.10, 4.20, 3.84, 3.43, 3.79, 3.74, 'c'),
    (207, 1.96, 3.00, 2.80, 2.59, 2.42, 1.61, 1.83, 1.21, 1.50, 'c'),
    (208, 2.70, 4.06, 3.98, 4.06, 3.93, 3.61, 2.91, 2.07, 2.67, 'c'),
    (209, 2.50, 4.37, 4.06, 3.68, 3.64, 3.17, 3.37, 3.20, 3.25, 'c'),
    (210, 2.35, 2.83, 2.79, 2.82, 2.79, 2.80, 2.76, 2.64, 2.69, 'c'),
    (211, 2.34, 4.06, 3.68, 3.59, 3.27, 2.60, 2.72, 2.22, 2.68, 'c'),
    (212, 2.20, 2.82, 1.90, 2.57, 2.30, 1.67, 1.90, 2.07, 1.76, 'c'),
    (214, 2.78, 3.18, 3.13, 3.11, 2.97, 3.06, 3.27, 3.24, 3.33, 'c'),
    (215, 3.43, 4.39, 4.63, 4.19, 4.00, 4.01, 3.66, 3.47, 3.22, 'c'),
    (216, 3.07, 3.90, 3.98, 4.09, 4.03, 4.07, 3.56, 3.83, 3.75, 'c'),
    (217, 1.21, 2.31, 2.19, 2.21, 2.09, 1.75, 1.72, 1.80, 1.36, 'c'),
    (218, 2.60, 3.19, 3.18, 3.15, 3.14, 3.08, 2.96, 2.97, 2.85, 'c'),
    (219, 2.61, 3.54, 3.45, 3.25, 3.01, 3.07, 2.65, 2.47, 2.55, 'c'),
    (220, 2.48, 2.99, 3.02, 3.02, 2.94, 2.69, 2.66, 2.68, 2.70, 'c'),
    (221, 3.73, 4.37, 4.20, 4.17, 4.19, 4.07, 3.86, 3.89, 3.89, 'c'),
    (222, 2.54, 3.26, 3.39, 3.27, 3.20, 3.32, 3.09, 3.25, 3.15, 'c'),
    (223, 2.83, 4.72, 4.97, 4.99, 4.96, 4.95, 4.82, 4.56, 4.49, 'c'),
    (224, 3.47, 4.27, 4.50, 4.34, 4.00, 4.11, 3.93, 3.68, 3.77, 'c'),
    (232, 2.79, 4.10, 3.85, 4.27, 4.01, 3.78, 3.14, 3.94, 3.69, 'c'),
    (201, 2.14, 2.36, 2.36, 2.28, 2.35, 2.31, 2.62, 2.12, 2.42, 'p'),
    (202, 3.37, 3.03, 3.02, 3.19, 2.98, 3.01, 2.75, 2.70, 2.84, 'p'),
    (203, 1.88, 1.99, 1.62, 1.65, 1.68, 1.65, 1.85, 1.96, 1.30, 'p'),
    (204, 3.10, 3.24, 3.37, 3.54, 3.31, 2.81, 3.58, 3.76, 3.05, 'p'),
    (205, 2.91, 3.35, 3.92, 3.69, 3.97, 3.94, 3.63, 2.92, 3.31, 'p'),
    (206, 2.29, 3.04, 3.28, 3.17, 2.99, 3.31, 3.21, 2.98, 2.82, 'p'),
    (207, 2.20, 2.46, 3.22, 2.65, 3.02, 2.25, 1.50, 2.37, 1.94, 'p'),
    (208, 2.70, 2.85, 2.81, 2.96, 2.69, 2.18, 1.91, 2.21, 1.71, 'p'),
    (209, 2.25, 3.45, 3.48, 3.80, 3.60, 2.83, 3.17, 3.22, 3.13, 'p'),
    (210, 2.48, 2.56, 2.52, 2.67, 2.60, 2.68, 2.64, 2.65, 2.61, 'p'),
    (211, 2.12, 2.19, 2.44, 2.41, 2.55, 2.93, 3.08, 3.11, 3.06, 'p'),
    (212, 2.37, 2.14, 1.92, 1.75, 1.58, 1.51, 1.94, 1.84, 1.76, 'p'),
    (214, 2.73, 2.57, 3.08, 2.62, 2.91, 2.71, 2.39, 2.42, 2.73, 'p'),
    (215, 3.15, 2.90, 2.80, 3.17, 2.39, 3.01, 3.22, 2.75, 3.14, 'p'),
    (216, 2.52, 3.02, 3.21, 3.17, 3.13, 3.38, 3.25, 3.29, 3.35, 'p'),
    (217, 1.48, 1.35, 1.15, 1.24, 1.32, 0.95, 1.24, 1.04, 1.16, 'p'),
    (218, 2.52, 2.61, 2.59, 2.77, 2.73, 2.70, 2.72, 2.71, 2.75, 'p'),
    (219, 2.90, 2.91, 2.89, 3.01, 2.74, 2.71, 2.86, 2.95, 2.66, 'p'),
    (220, 2.83, 2.78, 2.89, 2.77, 2.77, 2.69, 2.65, 2.84, 2.80, 'p'),
    (221, 3.50, 3.81, 3.77, 3.78, 3.90, 3.80, 3.78, 3.70, 3.61, 'p'),
    (222, 2.86, 3.06, 2.95, 3.07, 3.10, 2.67, 2.68, 2.94, 2.89, 'p'),
    (223, 2.42, 2.87, 3.08, 3.02, 3.14, 3.67, 3.84, 3.55, 3.75, 'p'),
    (224, 3.66, 3.98, 3.77, 3.65, 3.81, 3.77, 3.89, 3.63, 3.74, 'p'),
    (232, 2.88, 3.04, 3.00, 3.24, 3.37, 2.69, 2.89, 2.89, 2.76, 'p'),
]
columns = ['patient', 'basefev1', 'fev11h', 'fev12h', 'fev13h', 'fev14h', 'fev15h', 'fev16h', 'fev17h', 'fev18h', 'drug']
fev1 = pd.DataFrame(raw_data, columns=columns)

X = mt.create_design_matrix(fev1, 'drug')
Y = mt.create_response_matrix(fev1, ['fev11h', 'fev12h', 'fev13h', 'fev14h', 'fev15h', 'fev16h', 'fev17h', 'fev18h'])
L = mt.create_type_iii_hypothesis_contrast(X)
levels = np.array(np.arange(Y.shape[1])) + 1
degree = Y.shape[1] - 1
M = mt.create_orthopolynomial_contrasts(levels, degree)

results = mt.run_manova(X, Y, L, M)

print("L:\n", L)
print("M:\n", M)
print("b (aka beta hat):\n", results.b)
print("E (error SSCP):\n", results.E)
print("H (hypothesis SSCP):\n", results.H)
print(f"Wilks's Lambda: {results.wilks_lambda.statistic}; F value: {results.wilks_lambda.F}; Num DF: {results.wilks_lambda.df_n}; Den DF: {results.wilks_lambda.df_d}; Pr > F: {results.wilks_lambda.p_value}")
print(f"Pillai's Trace: {results.pillais_trace.statistic}; F value: {results.pillais_trace.F}; Num DF: {results.pillais_trace.df_n}; Den DF: {results.pillais_trace.df_d}; Pr > F: {results.pillais_trace.p_value}")
print(f"Hotelling-Lawley Trace: {results.hotelling_lawley_trace.statistic}; F value: {results.hotelling_lawley_trace.F}; Num DF: {results.hotelling_lawley_trace.df_n}; Den DF: {results.hotelling_lawley_trace.df_d}; Pr > F: {results.hotelling_lawley_trace.p_value}")
print(f"Roy's Largest Root: {results.roys_largest_root.statistic}; F value: {results.roys_largest_root.F}; Num DF: {results.roys_largest_root.df_n}; Den DF: {results.roys_largest_root.df_d}; Pr > F: {results.roys_largest_root.p_value}")
```

The results, which correspond to the “hour*drug” output of SAS, are as follows:

```
L:
 [[ 0. -1.  1.  0.]
 [ 0.  0. -1.  1.]]
M:
 [[-0.5401  0.5401  0.4308 -0.282   0.1498  0.0615  0.0171]
 [-0.3858  0.0772 -0.3077  0.5238 -0.4922 -0.3077 -0.1195]
 [-0.2315 -0.2315 -0.4308  0.1209  0.3638  0.5539  0.3585]
 [-0.0772 -0.3858 -0.1846 -0.3626  0.321  -0.3077 -0.5974]
 [ 0.0772 -0.3858  0.1846 -0.3626 -0.321  -0.3077  0.5974]
 [ 0.2315 -0.2315  0.4308  0.1209 -0.3638  0.5539 -0.3585]
 [ 0.3858  0.0772  0.3077  0.5238  0.4922 -0.3077  0.1195]
 [ 0.5401  0.5401 -0.4308 -0.282  -0.1498  0.0615 -0.0171]]
b (aka beta hat):
 [[2.4972 2.4783 2.4142 2.3397 2.2672 2.2194 2.1569 2.1495]
 [0.9916 0.9337 0.785  0.722  0.8016 0.7752 0.7269 0.7239]
 [1.1878 1.1421 1.1571 1.0978 0.9761 0.8598 0.814  0.8547]
 [0.3178 0.4025 0.4721 0.5199 0.4895 0.5844 0.616  0.5709]]
E (error SSCP):
 [[13.5301  0.0373  2.9558  1.2467 -0.2075 -0.1359  0.8915]
 [ 0.0373  3.372   0.6329  0.6544  0.0907  0.7994 -0.0791]
 [ 2.9558  0.6329  3.3866  1.2842 -0.2318  0.6817  0.4688]
 [ 1.2467  0.6544  1.2842  2.6505  1.0849 -0.3222  0.0958]
 [-0.2075  0.0907 -0.2318  1.0849  3.323  -0.4434 -0.7322]
 [-0.1359  0.7994  0.6817 -0.3222 -0.4434  2.3452 -0.2132]
 [ 0.8915 -0.0791  0.4688  0.0958 -0.7322 -0.2132  1.8829]]
H (hypothesis SSCP):
 [[ 5.0814 -0.8287  0.5302  0.7141  0.2897  0.1449 -0.4011]
 [-0.8287  0.4059  0.1986 -0.0068 -0.2869 -0.0562  0.1465]
 [ 0.5302  0.1986  0.3554  0.1899 -0.2221 -0.0192  0.0435]
 [ 0.7141 -0.0068  0.1899  0.1447 -0.0563  0.0072 -0.0236]
 [ 0.2897 -0.2869 -0.2221 -0.0563  0.2287  0.0371 -0.0946]
 [ 0.1449 -0.0562 -0.0192  0.0072  0.0371  0.0081 -0.0212]
 [-0.4011  0.1465  0.0435 -0.0236 -0.0946 -0.0212  0.0559]]
Wilks's Lambda: 0.511913190530635; F value: 3.578948797420777; Num DF: 14; Den DF: 126.0; Pr > F: 5.6675069777397695e-05
Pillai's Trace: 0.5549097973095972; F value: 3.5108265176304725; Num DF: 14.0; Den DF: 128.0; Pr > F: 7.191270361016035e-05
Hotelling-Lawley Trace: 0.8229204275679277; F value: 3.644361893515108; Num DF: 14.0; Den DF: 124.0; Pr > F: 4.5276826344189464e-05
Roy's Largest Root: 0.6083452743990968; F value: 5.5620139373631705; Num DF: 7; Den DF: 64; Pr > F: 4.939855356602183e-05
```

All four statistics are significant by any reasonable measure; the null hypothesis should be rejected.

### Box’s M test

The `perform_box_m_test` in MultivariaTeach can perform Box’s M test for the homogeneity of covariance matrices. It takes $\mathbf{X}$ and $\mathbf{Y}$ as prepared above and returns a $\chi^2$ statistic object similar to the $F$ statistic objects returned by the MANOVA test.

To perform the test, use code such as:

```python
BoxM = mt.perform_box_m_test(X, Y)
print(f"Box's M: {BoxM.statistic}; Chisq: {BoxM.chi2}; DF: {BoxM.df}; Pr > Chisq: {BoxM.p_value}")
```

In the case of Fisher’s Iris data, this will return:

```
Box's M: 146.6632492125118; Chisq: 140.94304992349774; DF: 20.0; Pr > Chisq: 0.0
```

Note again that MultivariaTeach returns the result of Python’s computation; in reporting these results, you should indicate that the calculated $p$-value is negligibly small, not zero. In the case of Box’s M Test, a small $p$-value directs you to reject the null hypothesis of equal of the covariance matrices for the dependent variables; in other words, there is significant evidence that the covariance matrices for the dependent variables in Fisher’s Iris data are different.

To calculate Box’s M Test, we first, with $\mathbf{S}_{\operatorname{pooled}}$ as the pooled sample covariance matrix, $\mathbf{S}_\ell$ the $\ell^{\textrm{th}}$ group sample covariance matrix, and $n_\ell$ as the sample size for the $\ell^{\textrm{th}}$ group, let
$$
M = \left( \sum_\ell (n_\ell-1) \right) \ln \left\lvert \mathbf{S}_{\operatorname{pooled}} \right\rvert - \sum_\ell \left( (n_\ell-1) \ln \left\lvert \mathbf{S}_\ell \right\rvert \right).
$$
Now, with $g$ as the number of groups and $p$ as the number of variables, let
$$
u = \left(\sum_\ell \frac{1}{n_\ell-1}-\frac{1}{\sum_\ell(n_\ell-1)}\right)\left( \frac{2p^2+3p-1}{6(p+1)(g-1)} \right).
$$
Then
$$
C = (1-u)M
$$
has an approximate $\chi^2$ distribution with $v = \frac{1}{2}p(p+1)(g-1)$ degrees of freedom.

The calculation is from page 310 of [Johnson & Wichern’s text](https://www.webpages.uidaho.edu/~stevel/519/Applied%20Multivariate%20Statistical%20Analysis%20by%20Johnson%20and%20Wichern.pdf).

### Mauchly’s Test of Sphericity

Mauchly’s test is used in repeated measures to test for equal variances among the differences between all possible pairings of the independent variables, aka “sphericity”. To perform the test with MultivariaTeach, you only need supply the $\mathbf{X}$ and $\mathbf{Y}$ matrices to the `mauchly` function; it will return the same type of $\chi^2$ statistic object as Box’s M test. Note that the test requires polynomial contrasts of a degree equal to the number of variables in $\mathbf{Y}$; since there are no other reasonable options for the $\mathbf{M}$ matrix used in the computation, the function automatically creates a suitable $\mathbf{M}$ matrix.

To perform Mauchly’s test:

```python
Mauchly = mt.mauchly(X, Y)
print(f"Mauchly's W: {Mauchly.statistic}; Chisq: {Mauchly.chi2}; DF: {Mauchly.df}; Pr > Chisq: {Mauchly.p_value}")
```

When performed on the FEV1 data above, the result is:
```
Mauchly's W: 0.0654899236469594; Chisq: 181.13981973529806; DF: 27.0; Pr > Chisq: 0.0
```

The vanishingly-small $p$-value — and, again, remember to not report it as actually being zero — indicates that we should reject the null hypothesis of sphericity; there is evidence of a difference in the variances between at least one pairing of independent variables.

Let $p$ be the number of variables and let $k=p-1$. To compute Mauchly’s Test, MultivariaTeach first creates $\mathbf{M}$ with $1, 2, \ldots, p$ levels with polynomials up to degree $k$. Then, with $\mathbf{S}_{\operatorname{pooled}}$ as usual, let
$$
W = \frac{\lvert \mathbf{M}^\intercal \mathbf{S}_{\operatorname{pooled}} \mathbf{M} \lvert}{(k^{-1}\operatorname{tr}(\mathbf{M}^\intercal \mathbf{S}_{\operatorname{pooled}} \mathbf{M}))^k}.
$$
Then let $n_1$ be the degrees of freedom, here calculated as the number of observations less the rank of $\mathbf{X}$ and let
$$
d = 1 - \frac{2k^2+k+2}{6kn_1}.
$$
Now, $-n_1d\ln(W)$ has an approximate $\chi^2$ distribution with $\frac{k(k+1)}{2}-1$ degrees of freedom.

### Greenhouse-Geisser correction

If Mauchly’s test indicates asphericity, you may wish to perform univariate tests with the Greenhouse-Geisser correction. The correction itself can be calculated with `mt.greenhouse_geisser_correction(Y, M)`, which will return the value of $\epsilon$ to apply to the univariate tests.

For example:

```python
epsilon = mt.greenhouse_geisser_correction(X, Y, M)
print(f"Greenhouse-Geisser's epsilon: {epsilon}")
```

With the FEV1 data, this produces:

```
Greenhouse-Geisser's epsilon: 0.49705792649184893
```

With perfect sphericity, $\epsilon = 1$; if sphericity does not hold at all, $\epsilon=0$. The low value (less than about 0.75) for $\epsilon$ for the FEV1 data is consistent with the rejection of the null hypothesis of sphericity we arrived at with Mauchly’s test.

MultivariaTeach calculates $\epsilon$ in one of two equivalent ways; one using the eigenvalues of $\mathbf{M}^\intercal \mathbf{S}_{\operatorname{pooled}} \mathbf{M}$, where $\mathbf{S}_{\operatorname{pooled}}$ is the pooled sample covariance; the other as defined in [Greenhouse and Geisser’s original 1959 paper in Vol. 24 No. 2 of Psychometrika](https://www.ece.uvic.ca/~bctill/papers/mocap/Greenhouse_Geisser_1959.pdf). Let $p$ be the number of variables; let $\boldsymbol{\Sigma} = \mathbf{S}_{\operatorname{pooled}}$ be the pooled sample covariance matrix; let $\sigma_{ts}$ be the elements of $\boldsymbol{\Sigma}$, let $\bar{\sigma}_{tt}$ be the mean of the diagonal elements of $\boldsymbol{\Sigma}$; let $\bar{\sigma}_{t.}$ be the mean of the $t^{\textrm{th}}$ row of $\boldsymbol{\Sigma}$; and let $\bar{\sigma}_{..}$ be the grand mean of $\boldsymbol{\Sigma}$. Then
$$
\epsilon = \frac{ p^2 (\bar{\sigma}_{tt} - \bar{\sigma}_{..}) }{ (p-1)\left( \sum \sum \sigma_{ts}^2 - 2p\sum \bar{\sigma}_{t.}^2 + p^2 \bar{\sigma}_{..}^2 \right) }.
$$
The eigenvalue method is the default; one may use the original method by passing the optional parameter `calculation_method="Original"`to the `greenhouse_geisser_correction()` function.

### Additional functions

#### Pooled Covariance

The function `calculate_pooled_covariance_matrix(X, Y)` will return $\mathbf{S}_{\operatorname{pooled}}$, the pooled covariance matrix used in many of the other calculations. For example, for Fisher’s Iris data:

```python
S_p = mt.calculate_pooled_covariance_matrix(X, Y)
print("S_pooled:\n", S_p)
```

The output:

```
S_pooled:
 [[0.265  0.0927 0.1675 0.0384]
 [0.0927 0.1154 0.0552 0.0327]
 [0.1675 0.0552 0.1852 0.0427]
 [0.0384 0.0327 0.0427 0.0419]]
```

#### Individual test statistic calculations

For each of the four main MANOVA test statistics, you can directly provide the $\mathbf{E}$ and $\mathbf{H}$ matrices in addition to the appropriate values for $p, q, v, s, m,$ and $n$ as described above.

These functions would be of most interest if you are supplied not with data to analyze but instead with the above information and asked to calculate the corresponding statistic. However, in such a circumstance, the math represented in the code may well be of more interest. In any case, the reader is strongly encouraged to examine the code directly should there be a need to directly perform these calculations.

## Testing

The `MultivariaTeach` package includes a suite of tests to ensure its functionality. To run the tests, you'll need to have `pytest` installed. If you don't have it installed, you can do so with the following command:

```[bash]
pip install pytest
```

After installing `pytest`, navigate to the directory containing the `MultivariaTeach` package, and run the following command:

```[bash]
pytest MultivariaTeach
```

This will execute the included tests and provide you with the results.

## Contributing

Implementation of additional multivariate statistical methods would be most welcome. Please [email](ben@trumpetpower.com) me with suggestions.

## License

This project is licensed under the ISC License. The full license text can be found in the [LICENSE](./LICENSE) file in the repository.

## Acknowledgments

Thanks first and foremost to Dr. Mark Reiser, who has been most generous of his knowledge and experience and patience in helping me to understand multivariate statistics.

Thanks also to ChatGPT, who has been a surprisingly good coach and even copilot. Though we've chased each other down rabbit holes, almost every rabbit hole has led to a deeper understanding.
