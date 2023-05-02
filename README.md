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

MultivariaTeach is designed to help students of multivariate statistics understand and perform a selection of fundamental tests. Where many statistical packages provide “helpful” features that automatically perform various functions or hide “useless” information, the goal of this package is to provide a selection of simple tools which can be easily enough put together as desired. For example, rather than use a command to indicate you wish to perform a repeated measures analysis, there is a tool to create a matrix of polynomial contrasts which you can then use in the rest of the analysis.

## Prerequisites

In addition to Python, you will need the standard `numpy`, `pandas`, and `scipy` packages installed (which is almost certainly already the case). Almost all of the mathematical calculations are done with `numpy`; the `pandas` library is used in the initial transformation of data into the necessary format; and `scipy` is used to calculate $p$-values.

It is assumed that you are a graduate-level student in a multivariate statistics course. The math won't make much sense to those without the necessary academic prerequisites; and, to those who have completed such a course, the more common statistical analysis software packages (such as SAS or R) are almost certainly more practical.

## Installation

To install the `multivariateach` package, simply run the following command:

```bash
pip install multivariateach
```

## Usage

### MANOVA

For those familiar with SAS, MultivariTeach provides much (but certainly not all!) of the basic functionality of `PROC GLM` — that is, of testing the null hypothesis $H_0: \mathbf{C} \boldsymbol{\beta} \mathbf{M} = \mathbf{0}$, against the alternative $H_a: \mathbf{C} \boldsymbol{\beta}\mathbf{M} \neq 0$, where $\mathbf{C}$ is a matrix of contrasts across varables; $\boldsymbol{\beta}$ is the parameter matrix; and $\mathbf{M}$ is a matrix of contrasts across groups.

In most software packages, performing a MANOVA test is done by loading all the data into an object and then writing out a math-looking command to indicate the test you wish to perform. For example, consider Ronald Fisher’s famous 1936 iris flower data set, with the species in the first column named `group` and the four measurements in the remaining columns, labeled `x1`, `x2`, `x3`, and `x4`. In Python, you can easily load the data as follows:

```python
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()
data = pd.DataFrame(np.hstack([iris.target.reshape(-1, 1), iris.data]))
data.columns = ['group', 'x1', 'x2', 'x3', 'x4']
```

Elsewhere, you might use something in the spirit of `manova data=data, test='group ~ x1 + x2 + x3 + x4'`. The software then ”helpfully” does this-and-that to the data and reports the various test statistics.

Here, instead, you must yourself prepare the $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{C}$, and $\mathbf{M}$ matrices which are used in the actual calcuations. Note that all software which performs a MANOVA does this, one way or another; it’s just that you mostly don’t ever get a chance to see how it happens. Further, there are different-but-equivalent mathematical approaches to take; here, we use the same ”reduced rank” model as SAS, but R uses the ”full rank” model instead.

In our model, $\mathbf{C}$ and $\mathbf{M}$ are as described above: matrices of contrasts. Here, $\mathbf{X}$ is the ”model matrix.” It has a leading column of ones, followed by ”dummy” indicators (either $1$ or $0$) that identify which group the observation (row) belongs to. Then $\mathbf{Y}$ is simply the observations themselves.

When you perform the MANOVA, the function first finds $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\intercal \mathbf{X})^- \mathbf{X}^\intercal \mathbf{Y}$, where $\mathbf{A}^\intercal$ denotes the transpose of $\mathbf{A}$ and $\mathbf{A}^-$ denotes the Moore-Penrose generalized inverse of $\mathbf{A}$. Then, the function calculates:
$$
\mathbf{E} = \mathbf{M}^\intercal(\mathbf{Y}^\intercal \mathbf{Y} - \mathbf{Y}^\intercal \mathbf{X} (\mathbf{X}^\intercal \mathbf{X})^- \mathbf{X}^\intercal \mathbf{Y})\mathbf{M}
$$
and
$$
\mathbf{B} = (\mathbf{C} \hat{\boldsymbol{\beta}})^\intercal (\mathbf{C} (\mathbf{X}^\intercal \mathbf{X}) \mathbf{C}^\intercal)^{-1} (\mathbf{C}{\hat{\boldsymbol{\beta}}}).
$$
Finally, $\mathbf{H} = \mathbf{M}^\intercal \mathbf{B} \mathbf{M}$. The $\mathbf{E}$ and $\mathbf{H}$ matrices are then used to calculate the test statistics.

While you are welcome to construct all four matrices by hand, MultivariaTeach provides tools to help construct them.

First, as usual, make sure you load the package at the top of your Python file:

```python
import MultivariaTeach as mt
```

This would typically be included with the lines above importing `pandas` and `sklearn`. We can now create our $\mathbf{X}$ and $\mathbf{Y}$ matrices:

```python
X = mt.create_design_matrix(data, 'group')
Y = mt.create_response_matrix(data, ['x1', 'x2', 'x3', 'x4'])
```

You can now inspect these two matrices, if you wish, before performing the analysis.

For this example, we will use a ”Type III” contrast matrix for $\mathbf{C}$. It is intended to test the null hypothesis of all variables having similar variance against the alternative of at least one pairing of variables having differing variance. Note that this hypothesis will not make sense in many real-world analyses; for example, it assumes that the variables are scaled comparably, and that they come from populations with similar variance, both of which are often not the case. However, it is the default analysis performed by SAS and is suitable for many textbook examples.

Creating such a matrix with MultivariaTeach is trivial:

```python
C = mt.create_contrast_type_iii(X)
```

In many MANOVA tests, there is no need to examine variation across groups, in which case one is testing $H_0: \mathbf{C} \boldsymbol{\beta} = \mathbf{0}$. To perform such a test in MultivariaTeach, use an identity matrix the size of the number of variables in $\mathbf{Y}$ for your $\mathbf{M}$ matrix. You can do this with `numpy`, another library which you should import at the top of your file:

```python
import numpy as np

M = np.eye(Y.shape[1])
```

With all the building blocks assembled, you can now perform the MANOVA:

```python
results = mt.run_manova(X, Y, C, M)
```

Unlike with most other software, you are not presented with a pretty-formatted table of results; instead, the `results` object can now be itself queried for … well, for anything and everything. So, for example, a simple-but-complete test might look like this:

```python
import numpy as np
import pandas as pd
from sklearn import datasets
import MultivariaTeach as mt

iris = datasets.load_iris() # Fisher's 1936 flower data

data = pd.DataFrame(np.hstack([iris.target.reshape(-1, 1), iris.data])) # Extract the parts we want in the structure we need
data.columns = ['group', 'x1', 'x2', 'x3', 'x4'] # Give names to the columns

X = mt.create_design_matrix(data, 'group')
Y = mt.create_response_matrix(data, ['x1', 'x2', 'x3', 'x4'])
C = mt.create_contrast_type_iii(X)
M = np.eye(Y.shape[1])

results = mt.run_manova(X, Y, C, M)

print("beta_hat: ", results.beta_hat)
print("E: ", results.E)
print("B: ", results.B)
print("H: ", results.H)
print("wilks_lambda: ", results.wilks_lambda.statistic, results.wilks_lambda.F, results.wilks_lambda.df_n, results.wilks_lambda.df_d, results.wilks_lambda.p_value)
print("pillais_trace: ", results.pillais_trace.statistic, results.pillais_trace.F, results.pillais_trace.df_n, results.pillais_trace.df_d, results.pillais_trace.p_value)
print("hotelling_lawley_trace: ", results.hotelling_lawley_trace.statistic, results.hotelling_lawley_trace.F, results.hotelling_lawley_trace.df_n, results.hotelling_lawley_trace.df_d, results.hotelling_lawley_trace.p_value)
print("roys_largest_root: ", results.roys_largest_root.statistic, results.roys_largest_root.F, results.roys_largest_root.df_n, results.roys_largest_root.df_d, results.roys_largest_root.p_value)
```

This will create the following output:

```[text]
beta_hat:  [[ 4.3825  2.293   2.8185  0.8995]
 [ 0.6235  1.135  -1.3565 -0.6535]
 [ 1.5535  0.477   1.4415  0.4265]
 [ 2.2055  0.681   2.7335  1.1265]]
E:  [[38.9562 13.63   24.6246  5.645 ]
 [13.63   16.962   8.1208  4.8084]
 [24.6246  8.1208 27.2226  6.2718]
 [ 5.645   4.8084  6.2718  6.1566]]
B:  [[ 63.21213333 -19.95266667 165.2484      71.27933333]
 [-19.95266667  11.34493333 -57.2396     -22.93266667]
 [165.2484     -57.2396     437.1028     186.774     ]
 [ 71.27933333 -22.93266667 186.774       80.41333333]]
H:  [[ 63.21213333 -19.95266667 165.2484      71.27933333]
 [-19.95266667  11.34493333 -57.2396     -22.93266667]
 [165.2484     -57.2396     437.1028     186.774     ]
 [ 71.27933333 -22.93266667 186.774       80.41333333]]
wilks_lambda:  0.023438630650877965 199.14534354008606 8 288 1.3650058325882981e-112
pillais_trace:  1.191898825041458 53.466488784612224 8.0 290.0 9.74216271945252e-53
hotelling_lawley_trace:  32.4773202409019 580.5320993061215 8.0 286.0 6.436176201217028e-172
roys_largest_root:  32.19192919827884 1166.957433437608 4 145 3.42202725324015e-19
```

With luck, by now the code above and its results are obvious and self-explanatory.

Note that, since $\mathbf{M} = \mathbf{I}$, our $\mathbf{B}$ and $\mathbf{H}$ matrices are the same for this example.

Also note that the $p$-values are given with their full precision — an hundred and twelve leading zeros in the case of Wilks’s Lambda above. As a general practice, when you report a $p$-value in your analysis, you should indicate it as being less than some arbitrarily small number, such as, “We find $p < 0.00001$ and therefore reject $H_0$.” If you use code to produce such output, be sure to check for vanishingly small $p$-values and adjust accordingly; MultivariaTeach will report the results of the calculation without adjustment.

### Repeated measures

One can use MANOVA with an $\mathbf{M}$ matrix of polynomial contrasts across time to perform an analysis of repeated measures. To construct such a matrix, use something such as the following:

```python
n = Y.shape[1] # number of contrasts
degree = 3 # up to cubic polynomials
M = orthopolynomial_cotrasts(n, degree)
```

The iris data, of course, is not a repeated measures experiment, so performing such a test with the data would not make sense.

### Box’s M test

The `perform_box_m_test` in MultivariaTeach can perform Box’s M test for the homogeneity of covariance matrices. It takes $\mathbf{X}$ and $\mathbf{Y}$ as prepared above and returns a $\chi^2$ statistic object similar to the $F$ statistic objects returned by the MANOVA test.

### Mauchly’s Test of Sphericity

Mauchly’s test is used in repeated measures to test for equal variances among the differences between all possible pairings of the independent variables. To perform the test with MultivariaTeach, you only need supply the $\mathbf{X}$ and $\mathbf{Y}$ matrices to the `mauchly` function; it will return the same type of $\chi^2$ statistic object as Box’s M test. Note that the test requires polynomial contrasts of a degree equal to the number of variables in $\mathbf{Y}$; since there are no other reasonable options for the $\mathbf{M}$ matrix used in the computation, the function automatically creates a suitable $\mathbf{M}$ matrix.

### Greenhouse-Geisser correction

If Mauchly’s test indicates asphericity, you may wish to perform univariate tests with the Greenhouse-Geisser correction. The correction itself can be calculated with `mt.greenhouse_geisser_correction(Y, M)`, which will return the value of $\epsilon$ to apply to the univariate tests. To perform the univariate tests, simply “pick apart” your $\mathbf{Y}$ matrix and analyze each column in a separate test. If you have a great many such columns … first, this package probably isn’t for you; but, if you insist, a simple loop iterating over the columns should make for cleaner code.

### Additional functions

#### Pooled Covariance

The function `calculate_pooled_covariance_matrix(X, Y)` will return $S_p$, the pooled covariance matrix used in many of the other calculations.

#### Individual test statistic calculations

For each of the four main MANOVA test statistics, you can directly provide the $\mathbf{E}$ and $\mathbf{H}$ matrices in addition to three or four of the following (depending on the statistic):

* n, the number of observations (rows of data);
* p, the number of variables (columns in $\mathbf{Y}$)
* g, the number of groups (equal to one less than the number of columns of $\mathbf{X}$)
* q, the rank of $\mathbf{C} (\mathbf{X}^\intercal \mathbf{X})^- \mathbf{C}^\intercal$.

These functions would be of most interest if you are supplied not with data to analyze but instead with the above information and asked to calculate the corresponding statistic. However, in such a circumstance, the math represented in the code may well be of more interest.

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

The math for the MANOVA test follows that specified by [SAS](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.3/statug/statug_introreg_sect038.htm).
