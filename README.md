# Causal Inference for Genomic Data with Multiple Heterogeneous Outcomes
This repository contains the code for reproducing simulation and real data analysis results of the paper "Causal Inference for Genomic Data with Multiple Heterogeneous Outcomes".


## Files


### Python module

- causarray: The main module for the proposed causal inference method.

### Scripts and Jupyter notebooks

- ex1: Simulation with Poisson DGP with sample splitting
    - `generate_simu_data.py`: Utility function that generates simulated data.
    - `run_simu.py`: Run simulation.
    - `simu.ipynb`: Plot the simulation results.
- ex2: LUHMES CRISPR data
    - `LUHMES.ipynb`: LUHMES data analysis.
- ex3: Lupus data
    - `Lupus.ipynb`: Lupus data analysis.


## Requirements

The following Python packages are required for the reproducibility workflow.


Package | Version
---|---
anndata | 0.9.2 
cvxpy | 1.1.18 
h5py | 3.1.0 
joblib | 1.1.0 
jupyter | 1.0.0
matplotlib | 3.4.3
numpy | 1.22.0 
pandas | 1.3.3 
python | 3.8.12
scanpy | 1.9.3 
scikit-learn | 1.1.2 
scipy | 1.10.1 
seaborn | 0.13.0
statsmodels | 0.13.5 
tqdm | 4.62.3





## Reproducibility workflow

For simulation studies, the workflow is as follows:

- Run script `run_simu.py` to perform simulation. The script takes two arguments: the simulation scenarios (0 for mean shifts and 1 for median shifts) and the number of folds (1 for no splitting and 5 for 5-fold cross-fitting). The results will be stored in the folder `result/simu/`.
- Use `simu.ipynb` to reproduce the figures (Figures 1 and E1-E2) based on the previous results.

For real data analysis on LUHMES data, the workflow is as follows:

- (Optional) Preprocessing
    - Store the [raw data file](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142078) as the folder `data/LUHMES/GSE142078_RAW` from the original paper [1].
    - Store the [supplemental code](https://genome.cshlp.org/content/suppl/2020/09/04/gr.262295.120.DC1) as the folder `data/LUHMES/Supplemental_Code` from the original paper [1].
    - Run `data/LUHMES/preprocess_data.R` to preprocess the LUHMES data, and the resulting data will be stored as `data/LUHMES/LUHMES.h5`, which is available in the current project folder.
- Use `LUHMES.ipynb` to run the proposed causal inference method and reproduce the figures (Figures 3-4 and G3-G7) and tables (Tables E1-E2).


For real data analysis on Lupus data, the workflow is as follows:

- Obtain the h5ad file of the lupus data from the authors of the original paper [2] and store it in the folder `data/lupus/GSE174188_CLUES1_adjusted.h5ad`.
- Run [`ex4_preprocess_lupus.py`](https://github.com/jaydu1/gcate/blob/main/ex4_preprocess_lupus.py) of the GitHub repo of paper [3] to preprocess the lupus data.
- Use `Lupus.ipynb` to run the proposed causal inference method and reproduce the figures (Figures 5-6, G3-G7, and E3).


# References

- [1] Lalli, M. A., Avey, D., Dougherty, J. D., Milbrandt, J., & Mitra, R. D. (2020). High-throughput single-cell functional elucidation of neurodevelopmental disease–associated genes reveals convergent mechanisms altering neuronal differentiation. Genome research, 30(9), 1317-1331.
- [2] Perez, Richard K., et al. "Single-cell RNA-seq reveals cell type–specific molecular and genetic associations to lupus." Science 376.6589 (2022): eabf1970.
- [3] Du, Jin-Hong, Larry Wasserman, and Kathryn Roeder. "Simultaneous inference for generalized linear models with unmeasured confounders." arXiv preprint arXiv:2309.07261 (2023).
