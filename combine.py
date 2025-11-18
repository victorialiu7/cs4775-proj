import pandas as pd

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
expr_samples = expr_genes.T
expr_samples.index.name = "PATIENT_ID"

# Find overlapping patients
expr_ids = set(expr_samples.index)
patient_ids = set(patient["PATIENT_ID"])
common_ids = sorted(expr_ids & patient_ids)

expr_samples = expr_samples.loc[common_ids]
patient_common = patient[patient["PATIENT_ID"].isin(common_ids)]

# Merge gene expression + patient data
merged = expr_samples.merge(
    patient_common,
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
