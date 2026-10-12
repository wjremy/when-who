# Software environment

Versions of the interpreter used for the analyses of paper v11 (recorded 11 October 2026):

| Software | Version |
|---|---|
| Python | 3.12.10 |
| NumPy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| numba (optional; speeds up the exact conditional calibration slope, results are identical without it) | 0.68.0 |
| statsmodels (optional; cross-check of the conditional slope in the tests) | 0.15.0 |
| R (for whenevalR) | 4.6.1, with testthat 3.3.2 |

`wheneval` itself needs only NumPy and pandas and runs on Python 3.9 or later. Install the pinned versions with `python -m pip install -r code/requirements.txt`, then the package with `python -m pip install ./code/wheneval`.
