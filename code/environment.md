# Reproducible software environment

- Python: 3.12.14 (the editor's analysis runtime; package metadata in `withinpatient/pyproject.toml` permits Python 3.9+).
- NumPy: 2.3.5; pandas: 3.0.1. These versions were recorded from the editor runtime on 6 October 2026.
- scikit-learn and Matplotlib: the installed `.dist-info` directories under `%LOCALAPPDATA%\Programs\Python\Python312\Lib\site-packages` could not be read in this environment (access denied), so their exact installed versions could not be verified. `requirements.txt` leaves these two dependencies unpinned rather than guessing.

Install the recorded numeric dependencies and the unpinned plotting/model dependencies with `python -m pip install -r code/requirements.txt`, then install the local package with `python -m pip install ./code/withinpatient`. For strict archival reproduction, record the actual scikit-learn and Matplotlib versions from the machine used for the run and pin them before execution.

Versions read from the interpreter used for all reported runs (6 Oct 2026): Python, numpy, pandas, scikit-learn, matplotlib = 3.12.10 2.5.3 3.0.6 1.9.1 3.11.2.
