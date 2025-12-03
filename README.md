# HMM-Based Gene Expression Risk Classifier

A Hidden Markov Model (HMM) implementation for predicting patient risk levels based on gene expression data from biological pathway gene sets.

## Overview

This project uses HMMs to model gene expression patterns and classify patients into high-risk and low-risk categories. The approach:

1. Computes module scores from gene sets (biological pathways)
2. Ranks modules by discriminative power
3. Discretizes continuous expression scores into observation states
4. Trains separate HMMs for high-risk and low-risk patient groups
5. Predicts risk by comparing likelihoods from both models

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Setup Instructions

1. **Clone the repository:**
```bash
git clone victorialiu7/cs4775-proj
cd cs4775-proj
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download and prepare the data:**

   **Step 4a: Download METABRIC data from cBioPortal**
   
   1. Go to [cBioPortal for Cancer Genomics](https://www.cbioportal.org/)
   2. Search for and select the **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium) study
   3. Download the complete dataset
   4. Unzip the downloaded file (`brca_metabric.tar.gz` or similar)
   5. From the extracted folder, copy these two files to your project directory:
      - `data_clinical_patient.txt` - Clinical patient data including risk labels
      - `data_mrna_illumina_microarray.txt` - Gene expression microarray data
   
   **Step 4b: Create merged dataset**
   
   You'll need to merge the clinical and expression data to create `merged.csv`. The merged file should have:
    - Gene expression columns (one per gene)
    - `PATIENT_ID` - Patient identifier
    - `high_risk` - Binary label (True/False) derived from clinical data
   
   **Step 4c: Gene sets**
   
    You need to create enrich-input1-GO_BP.tsv - Gene sets from GO Biological Process enrichment:

    Get seed gene list:
    - The paper uses a 322-gene list from "Consensus genes of the literature to predict breast cancer recurrence"
    - Alternative: Download the 41-gene CBCG list from CBCG website (https://cbcg.dk/causal.html)

    Perform enrichment analysis:
    - Go to GeneCodis (or similar tool like Enrichr, DAVID)
    - Input your gene list
    - Run GO Biological Process enrichment analysis
    - Download results as TSV

    Required format:
    - Tab-separated file with a genes column
    - Each row contains comma-separated gene names (e.g., "TP53, BRCA1, EGFR")
    - Filter to keep only gene sets with 2+ genes (to avoid zero standard deviation in calculations)

## Usage

Run the HMM classifier with 5-fold cross-validation:

```bash
python hmm.py
```

### Key Parameters

You can modify these in the code:

- `n_splits=5` - Number of cross-validation folds
- `n_states=6` - Number of hidden states in the HMM
- Gene set file path (currently `'enrich-input1-GO_BP.tsv'`)
- Data file path (currently `'merged.csv'`)

## Output

The script outputs:

- **Per-fold metrics:**
  - AUC (Area Under ROC Curve)
  - MCC (Matthews Correlation Coefficient)

- **Overall cross-validation metrics:**
  - Overall AUC
  - Overall MCC
  - Confusion matrix (TP, TN, FP, FN)
  - Accuracy, Sensitivity, Specificity

- **Top discriminative gene sets:**
  - Ranked by t-statistic

## Example Output

```
Number of gene sets: 150
Data shape: (200, 1502)
Class distribution:
False    120
True      80

============================================================
Fold 1/5
============================================================
Train size: 160 (High risk: 64)
Test size: 40 (High risk: 16)

Top 10 most discriminative gene sets:
  Module_1: 4.523
  Module_15: 3.891
  ...

Fold 1 Results:
  AUC: 0.7823
  MCC: 0.5234

============================================================
OVERALL CROSS-VALIDATION RESULTS
============================================================
Overall AUC: 0.7654
Overall MCC: 0.4987
...
```

## Data Sources

### METABRIC Dataset
This project uses the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset from cBioPortal:

- **Source:** [cBioPortal for Cancer Genomics](https://www.cbioportal.org/)
- **Study:** Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016)
- **Files required:**
  - `data_clinical_patient.txt` - Clinical and demographic information
  - `data_mrna_illumina_microarray.txt` - Gene expression profiles
- **Reference:** Curtis et al., Nature 2012; Pereira et al., Nat Commun 2016

### Gene Sets
- GO Biological Process gene sets from enrichment analysis
- Format: TSV file with `genes` column containing comma-separated gene symbols

### Module Score Calculation
For each gene set with genes g₁, g₂, ..., gₙ:

```
Module Score = √n × (mean expression) / (std expression)
```

### Observation State Discretization
Continuous module scores are discretized into n_states (default: 6) discrete observation symbols for the HMM.

### HMM Training
- Separate HMMs are trained for high-risk and low-risk patients
- Emission probabilities are estimated from training data
- Log-space computation prevents numerical underflow

### Prediction
For a test sample, compute log-likelihoods under both models and predict the class with higher likelihood.

## Key Fixes Implemented

The code includes several important fixes:

1. **Global gene set ranking** - Ranks modules once on all training data, not per-sample
2. **Proper observation sequences** - Discretizes continuous scores into HMM states
3. **Emission probability training** - Learns actual emission distributions
4. **Log-space computation** - Prevents numerical underflow in likelihood calculations
5. **No data leakage** - Gene set ranking and HMM training only use training fold data

## Project Structure

```
.
├── hmm.py                              # Main HMM classifier script
├── data_clinical_patient.txt           # METABRIC clinical data (not in repo)
├── data_mrna_illumina_microarray.txt   # METABRIC expression data (not in repo)
├── merged.csv                          # Processed merged dataset (not in repo)
├── enrich-input1-GO_BP.tsv            # Gene sets (not in repo)
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

**Note:** Data files are excluded from the repository per `.gitignore`. Users must download the METABRIC dataset from cBioPortal as described in Setup Instructions.
