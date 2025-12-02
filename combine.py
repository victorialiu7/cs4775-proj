import pandas as pd
import numpy as np

# Read gene expression matrix data
expr = pd.read_csv("data_mrna_illumina_microarray.txt", sep="\t")

# Read clinical patient data 
patient = pd.read_csv(
    "data_clinical_patient.txt",
    sep="\t",
    comment="#", # Skip metadata lines 
)

# Gene expression data table manipulation
expr_genes = expr.drop(columns=["Entrez_Gene_Id"]).set_index("Hugo_Symbol")

# Remove 0 variance genes 
gene_var = expr_genes.var(axis=1, skipna=True)
var_threshold = 1e-5
expr_genes = expr_genes.loc[gene_var > var_threshold]

# Remove genes with >15% missing
missing_fraction = expr_genes.isna().mean(axis=1)
expr_genes = expr_genes.loc[missing_fraction <= 0.15]

# Impute median per gene
expr_genes = expr_genes.apply(lambda row: row.fillna(row.median()), axis=1)

expr_samples = expr_genes.T
expr_samples.index.name = "PATIENT_ID"

# Find overlapping patients
expr_ids = set(expr_samples.index)
patient_ids = set(patient["PATIENT_ID"])
common_ids = sorted(expr_ids & patient_ids)

expr_samples = expr_samples.loc[common_ids]
patient_common = patient[patient["PATIENT_ID"].isin(common_ids)]

# Remove patients who received hormone therapy 
ht = patient_common["HORMONE_THERAPY"].astype(str).str.upper()
received_ht = ht.str.startswith("YES")
patient_no_ht = patient_common[~received_ht].copy()
no_ht_ids = set(patient_no_ht["PATIENT_ID"])
expr_samples = expr_samples.loc[sorted(expr_samples.index.intersection(no_ht_ids))]

# Z-score normalize 
means = expr_samples.mean(axis=0)
stds = expr_samples.std(axis=0, ddof=0)
expr_z = (expr_samples - means) / stds

# Arctan transformation
expr_atan = np.arctan(expr_z)
expr_final = (2 / np.pi) * expr_atan  # now roughly in [-1, 1]

# Merge gene expression + patient data
merged = expr_final.merge(
    patient_no_ht,
    left_index=True,
    right_on="PATIENT_ID",
    how="inner"
)

# Add high-risk label for patients with recurrence within 5 years
months = "RFS_MONTHS"
status = "RFS_STATUS"
has_recurred = merged[status].astype(str).str.startswith("1:")
merged["high_risk"] = (has_recurred) & (merged[months] <= 60)
merged["high_risk_label"] = merged["high_risk"].astype(int)

# Merged data exploration
print("\nMerged data shape:", merged.shape)
print("\nMerged data head:\n", merged.head())
print("\nLabel distribution:")
print(merged["high_risk_label"].value_counts())

# Save merged data to CSV
merged.to_csv("merged.csv", index=False)
