import json
import torch
import numpy as np
from general_utils import *
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import csv

def save_dataset_to_csv(ds, output_csv_path):
    """
    Save a HuggingFace Dataset to CSV using the question/answer logic.
    """
    rows = []
    for example in ds:
        rows.append({
            "question": str(example.get("question", "")).replace("\n", " ").strip(),
            "correct_answer": str(example.get("correct_answer", "")).replace("\n", " ").strip()
        })

    df = pd.DataFrame(rows)
    df.columns = df.columns.str.strip()  # clean headers
    df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\')

for dataset_name in ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']:
    for model in ['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it','Llama-3-8B-Instruct']:
        data_path = f"processed_data/{dataset_name}/prepared.json"
        dataset = load_dataset("json", data_files={"test": data_path})
        statistic_results = torch.load(f"results/{model}/{dataset_name}/{dataset_name}_features.pt", weights_only=False)
        hem_examples = statistic_results['hem_examples'] # hem score is after indices_to_remove operation
        indices_to_remove = statistic_results['indices_to_remove']
        ref_idx = statistic_results['ref_idx']
        remain_idx = statistic_results['remain_idx']

        right_threshold = 0.8
        wrong_threshold = 0.2
        y = np.array([1 if s > right_threshold else 0 for s in hem_examples])
        y_ref = y[ref_idx]
        y_remain = y[remain_idx]

        indices = np.arange(len(y_remain))

        temp_idx, test_idx, y_temp, y_test = train_test_split(
            indices, y_remain, test_size=0.2, random_state=42, stratify=y_remain)

        train_idx, val_idx, y_train, y_val = train_test_split(
            temp_idx, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        mask = [i not in indices_to_remove for i in range(len(dataset["test"]))]
        cleaned_data = dataset["test"].select([i for i, keep in enumerate(mask) if keep])
        ref_data = cleaned_data.select(ref_idx)
        remain_data = cleaned_data.select(remain_idx)
        train_data = remain_data.select(train_idx)
        val_data = remain_data.select(val_idx)
        test_data = remain_data.select(test_idx)

        save_dataset_to_csv(ref_data, f"processed_data/{dataset_name}/{model}_ref_data.csv")
        save_dataset_to_csv(train_data, f"processed_data/{dataset_name}/{model}_train_data.csv")
        save_dataset_to_csv(val_data, f"processed_data/{dataset_name}/{model}_val_data.csv")
        save_dataset_to_csv(test_data, f"processed_data/{dataset_name}/{model}_test_data.csv")

        labels = {'y_ref':y_ref, 'y_train':y_train, 'y_val':y_val, 'y_test':y_test}

        torch.save(labels,f"processed_data/{dataset_name}/{model}_labels.pt")

        indices = {'indices_to_remove':indices_to_remove, 'ref_idx': ref_idx,
            'remain_idx': remain_idx, 'train_idx': train_idx, 'val_idx':  val_idx, 'test_idx': test_idx}
        
        torch.save(indices,f"processed_data/{dataset_name}/{model}_indices.pt")

        print(model + "_" + dataset_name)

