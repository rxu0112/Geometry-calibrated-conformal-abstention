import numpy as np
from general_utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

def cp_threshold(scores_sorted, quantiles):
    """Return the empirical CP threshold (1-alpha coverage)"""
    n = len(scores_sorted)
    thresholds = []
    for level in quantiles:
        k = int(np.ceil((level) * (n + 1))) - 1  # -1 because Python index starts at 0
        k = min(k, n-1)  # avoid going out of bounds
        thresholds.append(scores_sorted[k])
    return np.array(thresholds)

def compute_conditioned_bound(idx,tau):

    # --- compute 1 - beta_corr
    # quantiles of correct-only calibration scores
    quantiles = np.linspace(0, 1, 1001)
    cal_corr_quantiles = cp_threshold(s_cal_corr_sorted, quantiles)
    cal_inc_quantiles = cp_threshold(s_cal_inc_sorted, quantiles)


    mask_corr = cal_corr_quantiles <= tau
    if not np.any(mask_corr):
        one_minus_beta_corr = 0.0
    else:
        one_minus_beta_corr = np.max(quantiles[mask_corr])

    mask_inc = cal_inc_quantiles <= tau
    if not np.any(mask_inc):
        one_minus_beta_inc = 0.0
    else:
        one_minus_beta_inc = np.max(quantiles[mask_inc])

    # theoretical interval
    lower_bound_corr = one_minus_beta_corr
    upper_bound_corr = one_minus_beta_corr + 1.0 / (len(s_cal_corr) + 1)
    lower_bound_inc = one_minus_beta_inc
    upper_bound_inc = one_minus_beta_inc + 1.0 / (len(s_cal_inc) + 1)

    lower_bound_corr = np.clip(lower_bound_corr, 0, 1)
    upper_bound_corr = np.clip(upper_bound_corr, 0, 1)
    lower_bound_inc = np.clip(lower_bound_inc, 0, 1)
    upper_bound_inc = np.clip(upper_bound_inc, 0, 1)
    lower_bound_all = retained_level[idx]
    upper_bound_all = lower_bound_all + 1.0 / (len(s_cal) + 1)
    lower_bound_cond_corr = lower_bound_corr/upper_bound_all*(len(s_test_corr)+len(s_cal_corr))/(len(s_test)+len(s_cal))
    ratio_corr = np.mean(s_test_corr <= tau)
    ratio_inc = np.mean(s_test_inc <= tau)
    ratio_all = np.mean(s_test <= tau)

    if (np.sum(s_test_corr <= tau)+np.sum(s_test_inc <= tau))==0:
        prob_cond_corr = 0.0
    else:
        prob_cond_corr = np.sum(s_test_corr <= tau)/(np.sum(s_test_corr <= tau)+np.sum(s_test_inc <= tau))
    
    lower_bounds_corr.append(lower_bound_corr)
    upper_bounds_corr.append(upper_bound_corr)
    lower_bounds_inc.append(lower_bound_inc)
    upper_bounds_inc.append(upper_bound_inc)
    lower_bounds_all.append(lower_bound_all)
    upper_bounds_all.append(upper_bound_all)
    lower_bounds_cond_corr.append(lower_bound_cond_corr)
    ratios_corr.append(ratio_corr)
    ratios_inc.append(ratio_inc)
    ratios_all.append(ratio_all)
    probs_cond_corr.append(prob_cond_corr)

calibrated_baseline = False
for dataset in ['nq','truthfulQA','sciq','simple_questions_wiki','common','gsm8k']: # 'nq','truthfulQA','sciq','simple_questions_wiki'
    for model in ['Llama-3-8B-Instruct','Gemma-3-4B-it','Llama-3.2-3B-Instruct']: # 'Gemma-3-4B-it','Llama-3.2-3B-Instruct'
        conformal_labels = np.load(f"results/{model}/{dataset}/test/label.npy")
        proposed = np.load(f"results/{model}/{dataset}/test/proposed.npy")

        if calibrated_baseline == False:
            AttentionScore = np.load(f"results/{model}/{dataset}/test/AttentionScore.npy")
            Perplexity = -np.load(f"results/{model}/{dataset}/test/Perplexity.npy") # negative perplexity for confidence
            SemanticEntropy = -np.load(f"results/{model}/{dataset}/test/SemanticEntropy.npy") # negative entropy for confidence
            SAR = -np.load(f"results/{model}/{dataset}/test/SAR.npy") # negative SAR for confidence
            EigenScore = -np.load(f"results/{model}/{dataset}/test/EigenScore.npy")
            Focus = -np.load(f"results/{model}/{dataset}/test/Focus.npy")
            PTrue = np.load(f"results/{model}/{dataset}/test/PTrue.npy")
            ATRMD = -np.load(f"results/{model}/{dataset}/test/ATRMD.npy")
        else:
            AttentionScore = np.load(f"results/{model}/{dataset}/test/AttentionScore_calibrated.npy")
            Perplexity = np.load(f"results/{model}/{dataset}/test/Perplexity_calibrated.npy") 
            SemanticEntropy = np.load(f"results/{model}/{dataset}/test/SemanticEntropy_calibrated.npy") 
            SAR = np.load(f"results/{model}/{dataset}/test/SAR_calibrated.npy")
            EigenScore = np.load(f"results/{model}/{dataset}/test/EigenScore_calibrated.npy")
            Focus = np.load(f"results/{model}/{dataset}/test/Focus_calibrated.npy")
            PTrue = np.load(f"results/{model}/{dataset}/test/PTrue_calibrated.npy")
        
        if calibrated_baseline == False:
            save_path = f"results/{model}/{dataset}/test/comparison.pkl"
        else:
            save_path = f"results/{model}/{dataset}/test/comparison_calibrated_baseline.pkl"

        # --- load existing results if available
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                comparison = pickle.load(f)
            computed_methods = set([entry['method'] for entry in comparison])
        else:
            comparison = []
            computed_methods = set()

        methods = [
            ("Proposed", proposed),
            ("Perplexity", Perplexity),
            ("SAR", SAR),
            ("SemanticEntropy", SemanticEntropy),
            ("Focus", Focus),
            ("AttentionScore", AttentionScore),
            ("EigenScore", EigenScore),
            ("P(True)", PTrue),
        ]

        if not calibrated_baseline:
            methods.append(("ATRMD", ATRMD))

        for method_name, confidences in methods:            
            if method_name in computed_methods:
                print(f"Skipping {method_name} (already computed)")
                continue

            print(f"Processing {model} {dataset} {method_name}")
            mask = np.isfinite(confidences)
            confidences = confidences[mask]
            conformal_labels_masked = conformal_labels[mask]
            auprc = average_precision_score(conformal_labels_masked, confidences)
            auroc = roc_auc_score(conformal_labels_masked, confidences)
            
            lower_bounds_corr_trials=[]
            upper_bounds_corr_trials=[]
            lower_bounds_inc_trials=[]
            upper_bounds_inc_trials=[]
            lower_bounds_all_trials=[]
            upper_bounds_all_trials=[]
            lower_bounds_cond_corr_trials=[]
            ratios_corr_trials = []
            ratios_inc_trials = []
            ratios_all_trials = []
            probs_cond_corr_trials = []
            auc_trials = []
            tau_trials = []
            
            for random_state in tqdm(list(range(1, 1001))):
                y_proba_cal, y_proba_test, y_cal, y_test = train_test_split(
                    confidences, conformal_labels_masked, random_state=random_state, test_size=0.5, stratify=conformal_labels_masked)

                s_cal = - y_proba_cal
                s_test = - y_proba_test
                s_cal_sorted = np.sort(s_cal)
                
                retained_level = np.arange(0.1, 1.00, 0.1)
                thresholds = cp_threshold(s_cal_sorted, retained_level)

                # correctly predicted calibration indices
                C_cal = np.where(y_cal == 1)[0]
                I_cal = np.where(y_cal == 0)[0]
                s_cal_corr = s_cal[C_cal]
                s_cal_inc = s_cal[I_cal]
                s_cal_corr_sorted = np.sort(s_cal_corr)
                s_cal_inc_sorted = np.sort(s_cal_inc)

                # --- empirical conditional probability on test
                C_test = np.where(y_test == 1)[0]
                I_test = np.where(y_test == 0)[0]
                s_test_corr = s_test[C_test]
                s_test_inc = s_test[I_test]

                lower_bounds_corr=[]
                upper_bounds_corr=[]
                lower_bounds_inc=[]
                upper_bounds_inc=[]
                lower_bounds_all=[]
                upper_bounds_all=[]
                lower_bounds_cond_corr=[]
                ratios_corr = []
                ratios_inc = []
                ratios_all = []
                probs_cond_corr = []

                for idx, tau in enumerate(thresholds):

                    compute_conditioned_bound(idx, tau)
                
                lower_bounds_corr_trials.append(lower_bounds_corr)
                upper_bounds_corr_trials.append(upper_bounds_corr)
                lower_bounds_inc_trials.append(lower_bounds_inc)
                upper_bounds_inc_trials.append(upper_bounds_inc)
                lower_bounds_all_trials.append(lower_bounds_all)
                upper_bounds_all_trials.append(upper_bounds_all)
                lower_bounds_cond_corr_trials.append(lower_bounds_cond_corr)
                ratios_corr_trials.append(ratios_corr)
                ratios_inc_trials.append(ratios_inc)
                ratios_all_trials.append(ratios_all)
                probs_cond_corr_trials.append(probs_cond_corr)
                auc_trials.append(np.trapz(probs_cond_corr, ratios_all))
                tau_trials.append(thresholds)

            comparison.append({
                'method': method_name,
                'LB retained correct': np.mean(lower_bounds_corr_trials, axis = 0),
                'UB retained correct': np.mean(upper_bounds_corr_trials, axis = 0),
                'LB retained incorrect': np.mean(lower_bounds_inc_trials, axis = 0),
                'UB retained incorrect': np.mean(upper_bounds_inc_trials, axis = 0),
                'LB retained all': np.mean(lower_bounds_all_trials, axis = 0),
                'UB retained all': np.mean(upper_bounds_all_trials, axis = 0),
                'LB conditional correct': np.mean(lower_bounds_cond_corr_trials, axis = 0),
                'retained correct': np.mean(ratios_corr_trials, axis = 0),
                'retained incorrect': np.mean(ratios_inc_trials, axis = 0),
                'retained all': np.mean(ratios_all_trials, axis = 0),
                'conditional correct': np.mean(probs_cond_corr_trials, axis=0),
                'AUC mean': np.mean(auc_trials),
                'AUC std': np.std(auc_trials),
                'AUPRC': auprc,
                'AUROC': auroc,
                'tau': np.mean(tau_trials, axis=0)})
            
            

            if calibrated_baseline == False:    
                with open(f"results/{model}/{dataset}/test/comparison.pkl", "wb") as f:
                    pickle.dump(comparison, f)
            else:
                with open(f"results/{model}/{dataset}/test/comparison_calibrated_baseline.pkl", "wb") as f:
                    pickle.dump(comparison, f)