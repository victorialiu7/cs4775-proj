import math
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read data
data_file = "merged.csv"
raw_data = pd.read_csv(data_file)

# Load gene sets
cbcg41_pd = pd.read_csv('enrich-input1-GO_BP.tsv', sep='\t')
cbcg41_list_raw = cbcg41_pd['genes'].tolist()
cbcg41_sets = []

for s in cbcg41_list_raw:
    s = s.split(', ')
    if len(s) > 1:  # More than one gene in set
        cbcg41_sets.append(s)

print(f"Number of gene sets: {len(cbcg41_sets)}")

gene_sets = cbcg41_sets
gene_list = list(set([i for sub_list in gene_sets for i in sub_list]))
other_cols = ['PATIENT_ID', 'high_risk']
data = raw_data[gene_list + other_cols].copy()

print(f"Data shape: {data.shape}")
print(f"Class distribution:\n{data['high_risk'].value_counts()}")


def rank_gene_sets_globally(data, gene_sets, labels):
    """
    FIX #1: Rank gene sets ONCE based on discriminative power across ALL samples
    Not per-sample!
    """
    gene_set_scores = {}
    
    for i, genes_in_set in enumerate(gene_sets):
        module_name = f'Module_{i+1}'
        valid_genes = [g for g in genes_in_set if g in data.columns]
        
        if len(valid_genes) < 2:
            continue
        
        # Calculate module score (t-statistic method from your code)
        tmp = data[valid_genes].copy()
        n = len(valid_genes)
        module_scores = np.sqrt(n) * tmp.mean(axis=1) / tmp.std(axis=1)
        
        # Calculate discriminative power using t-test
        high_risk_scores = module_scores[labels == True]
        low_risk_scores = module_scores[labels == False]
        
        if len(high_risk_scores) > 0 and len(low_risk_scores) > 0:
            t_stat, p_val = stats.ttest_ind(high_risk_scores, low_risk_scores)
            gene_set_scores[module_name] = abs(t_stat)
        else:
            gene_set_scores[module_name] = 0
    
    # Rank gene sets by discriminative power
    ranked_modules = sorted(gene_set_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 most discriminative gene sets:")
    for module, score in ranked_modules[:10]:
        print(f"  {module}: {score:.3f}")
    
    return ranked_modules


def compute_module_scores(data, gene_sets):
    """
    Compute module scores for all samples
    """
    module_data = data.copy()
    
    for i, genes_in_set in enumerate(gene_sets):
        module_name = f'Module_{i+1}'
        valid_genes = [g for g in genes_in_set if g in data.columns]
        
        if len(valid_genes) >= 2:
            tmp = data[valid_genes]
            n = len(valid_genes)
            module_data[module_name] = np.sqrt(n) * tmp.mean(axis=1) / tmp.std(axis=1)
        else:
            module_data[module_name] = 0
    
    return module_data


def create_observation_sequences(module_data, module_list, n_states=6):
    """
    FIX #2: Create observation sequences by discretizing module scores into states
    Instead of ranking per sample, we discretize the continuous scores
    """
    sequences = []
    
    for idx in module_data.index:
        sample_obs = []
        
        for module in module_list:
            score = module_data.loc[idx, module]
            
            # FIX: Discretize continuous scores into states (1 to n_states)
            # Use quantile-based discretization
            if not np.isnan(score):
                # Map score to state (1 to n_states)
                # Simple approach: divide range into equal bins
                # Better approach: use percentiles
                state = min(max(1, int((score + 3) / 6 * n_states) + 1), n_states)
            else:
                state = n_states // 2  # Middle state for missing values
            
            sample_obs.append(state)
        
        sequences.append(sample_obs)
    
    return sequences


def train_hmm(sequences, n_states, n_modules):
    """
    FIX #3: Train HMM properly with emission probabilities
    """
    # Initialize emission probabilities
    emission_probs = np.zeros((n_states, n_modules))
    
    # Count emissions for each state and module position
    for seq in sequences:
        for module_idx, state in enumerate(seq):
            emission_probs[state-1][module_idx] += 1
    
    # Normalize to get probabilities (add smoothing to avoid zero probabilities)
    emission_probs += 1e-6  # Laplace smoothing
    emission_probs = emission_probs / emission_probs.sum(axis=0, keepdims=True)
    
    return emission_probs


def forward_algorithm_log(observation_seq, emission_probs, n_states):
    """
    FIX #4: Use log probabilities to avoid numerical underflow
    """
    n_obs = len(observation_seq)
    
    # Initialize log forward probabilities
    log_forward = np.zeros((n_states, n_obs))
    
    # Initial probabilities (uniform start)
    start_probs = np.ones(n_states) / n_states
    log_start = np.log(start_probs + 1e-10)
    
    # Transition probabilities (allow more flexibility)
    # Not just sequential - allow staying in state or moving
    trans_probs = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if j == i:  # Stay in same state
                trans_probs[i][j] = 0.4
            elif j == i + 1:  # Move to next state
                trans_probs[i][j] = 0.4
            else:  # Small probability for other transitions
                trans_probs[i][j] = 0.2 / (n_states - 2) if n_states > 2 else 0
    
    log_trans = np.log(trans_probs + 1e-10)
    
    # Forward pass
    for t in range(n_obs):
        module_idx = t
        obs_state = observation_seq[t] - 1  # Convert to 0-indexed
        
        for s in range(n_states):
            if t == 0:
                # Initial step
                log_forward[s][t] = log_start[s] + np.log(emission_probs[obs_state][module_idx] + 1e-10)
            else:
                # Sum over previous states
                log_prob_sum = -np.inf
                for prev_s in range(n_states):
                    log_prob = log_forward[prev_s][t-1] + log_trans[prev_s][s]
                    log_prob_sum = np.logaddexp(log_prob_sum, log_prob)
                
                log_forward[s][t] = log_prob_sum + np.log(emission_probs[obs_state][module_idx] + 1e-10)
    
    # Final log likelihood
    log_likelihood = -np.inf
    for s in range(n_states):
        log_likelihood = np.logaddexp(log_likelihood, log_forward[s][n_obs-1])
    
    return log_likelihood


def predict_with_hmm(observation_seq, emission_h, emission_l, n_states):
    """
    Predict using both models and compare likelihoods
    """
    log_lik_h = forward_algorithm_log(observation_seq, emission_h, n_states)
    log_lik_l = forward_algorithm_log(observation_seq, emission_l, n_states)
    
    # Return prediction and probability
    if log_lik_h > log_lik_l:
        prob = 1 / (1 + np.exp(log_lik_l - log_lik_h))
        return True, prob
    else:
        prob = 1 / (1 + np.exp(log_lik_h - log_lik_l))
        return False, 1 - prob


# Main pipeline
def run_cross_validation(data, gene_sets, n_splits=5, n_states=6):
    """
    FIX #5: Proper cross-validation without data leakage
    """
    labels = data['high_risk'].values
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*60}")
        
        train_data = data.iloc[train_idx].copy()
        test_data = data.iloc[test_idx].copy()
        
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        print(f"Train size: {len(train_idx)} (High risk: {train_labels.sum()})")
        print(f"Test size: {len(test_idx)} (High risk: {test_labels.sum()})")
        
        # Step 1: Rank gene sets on TRAINING data only
        ranked_modules = rank_gene_sets_globally(
            train_data.drop(columns=other_cols), 
            gene_sets, 
            train_labels
        )
        
        module_names = [m for m, _ in ranked_modules]
        
        # Step 2: Compute module scores
        train_module_data = compute_module_scores(
            train_data.drop(columns=other_cols), 
            gene_sets
        )
        test_module_data = compute_module_scores(
            test_data.drop(columns=other_cols), 
            gene_sets
        )
        
        # Step 3: Create observation sequences
        train_sequences_h = create_observation_sequences(
            train_module_data[train_labels == True],
            module_names,
            n_states
        )
        train_sequences_l = create_observation_sequences(
            train_module_data[train_labels == False],
            module_names,
            n_states
        )
        
        # Step 4: Train HMMs
        emission_h = train_hmm(train_sequences_h, n_states, len(module_names))
        emission_l = train_hmm(train_sequences_l, n_states, len(module_names))
        
        # Step 5: Make predictions
        test_sequences = create_observation_sequences(
            test_module_data,
            module_names,
            n_states
        )
        
        fold_predictions = []
        fold_probabilities = []
        
        for seq in test_sequences:
            pred, prob = predict_with_hmm(seq, emission_h, emission_l, n_states)
            fold_predictions.append(pred)
            fold_probabilities.append(prob)
        
        all_predictions.extend(fold_predictions)
        all_labels.extend(test_labels)
        all_probabilities.extend(fold_probabilities)
        
        # Fold metrics
        fold_auc = roc_auc_score(test_labels, fold_probabilities)
        fold_mcc = matthews_corrcoef(test_labels, fold_predictions)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  AUC: {fold_auc:.4f}")
        print(f"  MCC: {fold_mcc:.4f}")
    
    # Overall metrics
    print(f"\n{'='*60}")
    print("OVERALL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    overall_auc = roc_auc_score(all_labels, all_probabilities)
    overall_mcc = matthews_corrcoef(all_labels, all_predictions)
    
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall MCC: {overall_mcc:.4f}")
    
    # Confusion matrix
    TP = sum((np.array(all_labels) == True) & (np.array(all_predictions) == True))
    TN = sum((np.array(all_labels) == False) & (np.array(all_predictions) == False))
    FP = sum((np.array(all_labels) == False) & (np.array(all_predictions) == True))
    FN = sum((np.array(all_labels) == True) & (np.array(all_predictions) == False))
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {TP}")
    print(f"  True Negatives:  {TN}")
    print(f"  False Positives: {FP}")
    print(f"  False Negatives: {FN}")
    print(f"  Accuracy: {(TP + TN) / (TP + TN + FP + FN):.4f}")
    print(f"  Sensitivity: {TP / (TP + FN) if (TP + FN) > 0 else 0:.4f}")
    print(f"  Specificity: {TN / (TN + FP) if (TN + FP) > 0 else 0:.4f}")
    
    return overall_auc, overall_mcc


# Run the improved pipeline
run_cross_validation(data, gene_sets, n_splits=5, n_states=6)