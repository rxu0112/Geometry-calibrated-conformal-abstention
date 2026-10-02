import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from general_utils import *
from tqdm import tqdm
# Compute a dimension-matched PCA baseline from hidden states.

for dataset in ['nq', 'sciq', 'common','simple_questions_wiki', 'truthfulQA', 'gsm8k']:
    for model in ['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct']:
        print(model + " " + dataset)
        right_threshold = 0.8
        wrong_threshold = 0.2
        ref_size = 0.5
        if dataset == 'truthfulQA':
            ref_size = 0.45

        if dataset in ['sciq', 'nq', 'simple_questions_wiki', 'truthfulQA','gsm8k']:
            with open(f"results/{model}/{dataset}/{dataset}_gpt-4o_judge.json", "r") as f:
                    judge_examples_gpt4o = json.load(f)

            with open(f"results/{model}/{dataset}/{dataset}_gpt-5.1_judge.json", "r") as f:
                    judge_examples_gpt5 = json.load(f)

        if dataset in ['truthfulQA', 'nq']:
            hem_examples = []
            for idx in range(len(judge_examples_gpt4o)):
                if judge_examples_gpt4o[idx] in ["right","Right"] and judge_examples_gpt5[idx] in ["right","Right"]:
                    hem_examples.append(1.0)
                elif judge_examples_gpt4o[idx] in ["wrong","Wrong"] and judge_examples_gpt5[idx] in ["wrong","Wrong"]:
                    hem_examples.append(0.0)
                else:
                    hem_examples.append(0.5)
            word_limit = 50
        elif dataset in ['sciq','simple_questions_wiki','gsm8k']:
            hem_examples = torch.load(f"results/{model}/{dataset}/{dataset}_hhem_judge.pt", weights_only=False)
            judge_idx = 0
            for hem_idx in range(len(hem_examples)):
                if hem_examples[hem_idx] < 0.2:
                    if judge_examples_gpt4o[judge_idx] in ["right","Right"] and judge_examples_gpt5[judge_idx] in ["right","Right"]:
                        hem_examples[hem_idx] = 1.0
                    elif judge_examples_gpt4o[judge_idx] in ["wrong","Wrong"] and judge_examples_gpt5[judge_idx] in ["wrong","Wrong"]:
                        hem_examples[hem_idx] = 0.0
                    else:
                        hem_examples[hem_idx] = 0.5
                    judge_idx += 1
            word_limit=10
        elif dataset in ['common']:
            hem_examples = torch.load(f"results/{model}/{dataset}/{dataset}_hhem_judge.pt", weights_only=False)
            word_limit=10


        save_path = f'results/{model}/{dataset}/{dataset}_pca_features.pt'
        lens_results = load_pickle_file(f"results/{model}/{dataset}/{dataset}_lens_results.pkl")
        prob_history_examples = sum([chunk['prob_history_examples'] for chunk in lens_results], [])
        lens_token_examples = sum([chunk['generated_tokens_examples'] for chunk in lens_results], [])
        h_last_history_examples = sum([chunk['h_last_history_examples'] for chunk in lens_results], [])

        indices_to_remove = []
        for s_idx, gen in enumerate(lens_token_examples):
            remove = False

            # 0. Empty generation
            if len(gen) == 0:
                remove = True

            # 1. Text-based filtering - dataset specific
            if dataset in ['truthfulQA', 'nq']:
                if len(gen) >= word_limit:
                    remove = True
            
            if dataset in ['sciq', 'simple_questions_wiki','common']:
                if any(k in gen for k in ("\n", "incorrect", "Question", "No", "correct", "?")) \
                or len(gen) >= word_limit \
                or (len(gen)>0 and gen[0] == " "):
                    remove = True

            if dataset in ['gsm8k']:
                if any(k in gen for k in ("\n", "incorrect", "Question", "No", "correct", "?")) \
                    or len(gen) >= word_limit:
                    remove = True

            if gen != lens_token_examples[s_idx]:
                remove = True
                print(s_idx)

            # 2. HEM score filtering
            if wrong_threshold <= hem_examples[s_idx] <= right_threshold:
                remove = True
            
            if remove:
                indices_to_remove.append(s_idx)


        prob_history_examples = [v for i, v in enumerate(prob_history_examples) if i not in indices_to_remove]
        hem_examples = [v for i, v in enumerate(hem_examples) if i not in indices_to_remove]
        h_last_history_examples = [v for i, v in enumerate(h_last_history_examples) if i not in indices_to_remove]

        # Reference/remain split
        y = np.array([1 if s > right_threshold else 0 for s in hem_examples])
        indices = np.arange(len(y))

        remain_idx, ref_idx, y_remain, y_ref = train_test_split(
            indices, y, test_size=ref_size, random_state=42, stratify=y)
        ref_idx, fit_idx, y_ref, y_fit = train_test_split(ref_idx, y_ref, test_size=0.5, random_state=42)

        num_hidden_states = len(h_last_history_examples[0][0])

        if num_hidden_states % 2 != 0:
            raise ValueError("Expected two hidden states per Transformer layer.")

        num_layers = num_hidden_states // 2
        shared_fit_hidden = []

        for s_idx in fit_idx:
            for t_idx in range(len(h_last_history_examples[s_idx])):
                h_last_history = h_last_history_examples[s_idx][t_idx]

                for l_idx in range(num_layers):
                    layer_hidden = np.concatenate([
                        np.asarray(h_last_history[2 * l_idx]).reshape(-1),
                        np.asarray(h_last_history[2 * l_idx + 1]).reshape(-1),
                    ]).astype(np.float32, copy=False)

                    shared_fit_hidden.append(layer_hidden)

        shared_fit_data = np.stack(shared_fit_hidden)
        if len({row.shape[0] for row in shared_fit_hidden}) != 1:
            raise ValueError(
                "All layers must have the same hidden dimension."
            )

        shared_pca = PCA(
            n_components=8,
            svd_solver="arpack",
            tol=1.0,
            random_state=42,
        )
        shared_pca.fit(shared_fit_data)
        
        # Transform every token with the frozen layer-wise PCA models.
        prob_examples = []
        pca_hidden_examples = []

        for s_idx in tqdm(range(len(h_last_history_examples))):
            prob_tokens = []
            pca_hidden_tokens = []

            for t_idx in range(len(h_last_history_examples[s_idx])):
                prob_history = prob_history_examples[s_idx][t_idx]
                prob_tokens.append(prob_history[-1])

                h_last_history = h_last_history_examples[s_idx][t_idx]

                pca_hidden = []

                for l_idx in range(num_layers):
                    layer_hidden = np.concatenate([
                        np.asarray(h_last_history[2 * l_idx]).reshape(-1),
                        np.asarray(h_last_history[2 * l_idx + 1]).reshape(-1),
                    ]).astype(np.float32, copy=False)

                    projected = shared_pca.transform(
                        layer_hidden.reshape(1, -1)
                    )[0]

                    # Preserve the original dimension: 8 per layer, 4 for the last.
                    output_dim = 4 if l_idx == num_layers - 1 else 8
                    pca_hidden.extend(projected[:output_dim])

                pca_hidden = np.asarray(pca_hidden)

                expected_dim = 8 * num_layers - 4
                assert pca_hidden.shape == (expected_dim,)

                pca_hidden_tokens.append(np.asarray(pca_hidden))

            prob_examples.append(np.array(prob_tokens))
            pca_hidden_examples.append(np.stack(pca_hidden_tokens, axis=1))

        statistics = {'hem_examples': hem_examples,
                    'prob_examples': prob_examples,
                    'pca_hidden_examples': pca_hidden_examples,
                    'indices_to_remove': indices_to_remove,
                    'ref_idx': ref_idx,
                    'remain_idx': remain_idx,
        }

        torch.save(statistics,save_path)

