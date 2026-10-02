import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from general_utils import *
from tqdm import tqdm
# set a split reference set, cos to an average hidden state

metric = 'manhattan'

for dataset in  ['sciq', 'nq', 'common', 'simple_questions_wiki', 'truthfulQA', 'gsm8k']:
    for model in ['Llama-3.2-3B-Instruct', 'Llama-3-8B-Instruct', 'Gemma-3-4B-it']:
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


        save_path = f'results/{model}/{dataset}/{dataset}_features.pt'
        lens_results = load_pickle_file(f"results/{model}/{dataset}/{dataset}_lens_results.pkl")
        prob_history_examples = sum([chunk['prob_history_examples'] for chunk in lens_results], [])
        lens_token_examples = sum([chunk['generated_tokens_examples'] for chunk in lens_results], [])
        h_last_history_examples = sum([chunk['h_last_history_examples'] for chunk in lens_results], [])
        contri_chunks = load_pickle_file(f"results/{model}/{dataset}/{dataset}_contri_results_{metric}.pkl")
        contri_results = [x for chunk in contri_chunks for x in chunk]

        indices_to_remove = []
        indices_to_remove_text = []
        for s_idx, output in enumerate(contri_results):
            gen = output[1]["generated tokens"]
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

            # 2. Length/token mismatch
            if gen != lens_token_examples[s_idx]:
                remove = True
                print(s_idx)

            # 3. HEM score filtering
            if wrong_threshold <= hem_examples[s_idx] <= right_threshold:
                remove = True
        
            if remove:
                indices_to_remove.append(s_idx)


        results = [v for i, v in enumerate(contri_results) if i not in indices_to_remove]
        prob_history_examples = [v for i, v in enumerate(prob_history_examples) if i not in indices_to_remove]
        hem_examples = [v for i, v in enumerate(hem_examples) if i not in indices_to_remove]
        h_last_history_examples = [v for i, v in enumerate(h_last_history_examples) if i not in indices_to_remove]

        # reference exhibition
        y = np.array([1 if s > right_threshold else 0 for s in hem_examples])
        indices = np.arange(len(y))

        remain_idx, ref_idx, y_remain, y_ref = train_test_split(
            indices, y, test_size=ref_size, random_state=42, stratify=y)

        ref_right_hidden = defaultdict(list)
        ref_wrong_hidden = defaultdict(list)

        for s_idx, output in enumerate(results):
            if s_idx in ref_idx:
                target_hidden = ref_right_hidden if hem_examples[s_idx] > right_threshold else ref_wrong_hidden
                for t_idx in range(len(output[1]["generated tokens"])):
                    h_last_history = h_last_history_examples[s_idx][t_idx]
                    for l_idx, h in enumerate(h_last_history):
                        target_hidden[l_idx].append(h)

        ref_right_hidden_avg = defaultdict(list)
        ref_wrong_hidden_avg = defaultdict(list)

        for l_idx in range(len(ref_right_hidden)):
            ref_right_hidden_avg[l_idx]=np.mean(ref_right_hidden[l_idx], axis=0)
            ref_wrong_hidden_avg[l_idx]=np.mean(ref_wrong_hidden[l_idx], axis=0)

        # metric computation
        contri_examples = []
        prob_diff_examples = []
        angle_diff_examples = []
        prob_examples = []
        cos_to_right_hidden_avg_examples = []
        cos_to_wrong_hidden_avg_examples = []

        for s_idx, output in enumerate(tqdm(results)):

            contri_tokens = []
            prob_diff_tokens = []
            angle_diff_tokens = []
            prob_tokens = []
            cos_to_right_hidden_avg_tokens = []
            cos_to_wrong_hidden_avg_tokens = []

            for t_idx in range(len(output[1]['generated tokens'])):
                att_contributions = output[1]['contrib mats loop'][t_idx][0]                
                mlp_con_matrices = output[1]['contrib mats loop'][t_idx][1]
                mlp_res_matrices = output[1]['contrib mats loop'][t_idx][2]

                mlp_con_matrices = np.stack(mlp_con_matrices)
                proceeding_att = []
                for l_idx in range(len(att_contributions)):
                    proceeding_att.append(att_contributions[l_idx][0, :])
                proceeding_mlp = []
                for l_idx in range(mlp_con_matrices.shape[0]):
                    if l_idx == 0:
                        proceeding_mlp.append(np.zeros(mlp_con_matrices.shape[1]))
                    proceeding_mlp.append(mlp_con_matrices[l_idx - 1, :].copy())
                for l_idx in range(len(proceeding_mlp)):
                    proceeding_mlp[l_idx][-1] = 0

                last_mlp = mlp_con_matrices.copy()
                last_mlp[:, :-1] = 0

                mlp_to_logit = []
                for l_idx in range(len(att_contributions) * 2):
                    if l_idx % 2 == 0:
                        mlp_to_logit.append(proceeding_att[int(l_idx / 2)] * proceeding_mlp[int(l_idx / 2)])
                    else:
                        mlp_to_logit.append(last_mlp[int((l_idx + 1) / 2) - 1, :])
                mlp_to_logit = np.stack(mlp_to_logit)

                proceeding_contri = np.sum(mlp_to_logit[:, :-1], axis=1)
                last_contri = mlp_to_logit[:, -1]

                contri = proceeding_contri + last_contri
                contri_tokens.append(contri[1:])

                prob_history = prob_history_examples[s_idx][t_idx]
                prob_diff = [prob_history[l_idx + 1] - prob_history[l_idx] for l_idx in range(len(prob_history) - 1)]
                prob_diff_tokens.append(prob_diff)
                prob_tokens.append(prob_history[-1])

                h_last_history = h_last_history_examples[s_idx][t_idx]
                cosin_similarity_diff = [cos_np(h_last_history[l_idx + 1], h_last_history[l_idx]) for l_idx in range(len(h_last_history) - 1)]
                angle_diff = [np.arccos(np.clip(c, -1.0, 1.0)) for c in cosin_similarity_diff]
                angle_diff_tokens.append(angle_diff)

                cos_to_right_hidden_avg = [cos_np(h_last_history[l_idx],ref_right_hidden_avg[l_idx]) for l_idx in range(len(h_last_history))]
                cos_to_right_hidden_avg_tokens.append(cos_to_right_hidden_avg)

                cos_to_wrong_hidden_avg = [cos_np(h_last_history[l_idx],ref_wrong_hidden_avg[l_idx]) for l_idx in range(len(h_last_history))]
                cos_to_wrong_hidden_avg_tokens.append(cos_to_wrong_hidden_avg)

            contri_examples.append(np.stack(contri_tokens,axis=1))
            prob_diff_examples.append(np.stack(prob_diff_tokens,axis=1))
            angle_diff_examples.append(np.stack(angle_diff_tokens,axis=1))
            prob_examples.append(np.array(prob_tokens))
            cos_to_right_hidden_avg_examples.append(np.stack(cos_to_right_hidden_avg_tokens,axis=1))
            cos_to_wrong_hidden_avg_examples.append(np.stack(cos_to_wrong_hidden_avg_tokens,axis=1))

        statistics = {'hem_examples': hem_examples,
                    'prob_examples': prob_examples,
                    'contri_examples':  contri_examples,
                    'angle_diff_examples': angle_diff_examples,
                    'prob_diff_examples': prob_diff_examples,
                    'cos_to_right_hidden_avg_examples': cos_to_right_hidden_avg_examples,
                    'cos_to_wrong_hidden_avg_examples': cos_to_wrong_hidden_avg_examples,
                    'indices_to_remove': indices_to_remove,
                    'ref_idx': ref_idx,
                    'remain_idx': remain_idx,
        }

        torch.save(statistics,save_path)

