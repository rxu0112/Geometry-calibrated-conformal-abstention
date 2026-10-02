import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import Focus
from lm_polygraph.utils import estimate_uncertainty
from lm_polygraph.utils.generation_parameters import GenerationParameters
import argparse
import pickle
import os
import sys
from tqdm import tqdm
from pathlib import Path
import lm_polygraph.estimators.focus as focus_module

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# Monkey patch the calcu_idf function to support dataset configs
def patched_calcu_idf(
    tokenizer_path, path, idf_dataset, trust_remote_code, idf_seed, idf_dataset_size
):

    import math
    import random
    from collections import defaultdict
    from transformers import AutoTokenizer

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if "/" in idf_dataset and idf_dataset.count("/") == 1:
        #"dataset_name/config_name"
        dataset_name, config_name = idf_dataset.rsplit("/", 1)
        if "." in config_name or config_name.startswith("wikitext"):
            dataset = load_dataset(dataset_name, config_name, trust_remote_code=trust_remote_code)
        else:
            dataset = load_dataset(idf_dataset, trust_remote_code=trust_remote_code)
    else:
        try:
            dataset = load_dataset(idf_dataset, trust_remote_code=trust_remote_code)
        except ValueError:
            print(f"warning: dataset {idf_dataset} needs to be configurated, trying to wikitext-103-raw-v1")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    data = [d for d in dataset["train"]]
    rng = random.Random(idf_seed)
    rng.shuffle(data)

    if (idf_dataset_size > 0) and (idf_dataset_size < len(data)):
        data = rng.sample(data, idf_dataset_size)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    document_frequency = defaultdict(int)
    offset = 1 if "facebook" in tokenizer_path else 0
    for doc in tqdm(data, desc="compute IDF"):
        tokenized_doc = tokenizer(doc["text"])["input_ids"][offset:]
        unique_tokens = set(tokenized_doc)
        for token in unique_tokens:
            document_frequency[token] += 1

    total_documents = len(data)
    pickle.dump(
        np.array(
            [
                math.log(total_documents / (document_frequency[i] + 1))
                for i in range(len(tokenizer.vocab))
            ]
        ),
        open(path, "wb"),
    )

# 应用补丁
focus_module.calcu_idf = patched_calcu_idf


def load_pickle_file(path):
    objs = []
    with open(path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break
    return objs


def load_model_and_tokenizer(model_name, device):
    model_paths = {
        'Llama-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct',
        'Gemma-3-4B-it': 'google/gemma-3-4b-it',
        'Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    }

    print(f"loading model {model_name}")
    model_path = model_paths[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name in ['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct']:
        tokenizer.pad_token_id = 128001
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map=device)
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if model_name == 'Gemma-3-4B-it':
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, attn_implementation="eager", device_map=device)

    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()

    return base_model, tokenizer


def load_dataset_and_split(dataset_name, model_name, mode, base_path="processed_data"):

    print(f"loading dataset: {dataset_name}")

    csv_path = f"{base_path}/{dataset_name}/{model_name}_{mode}_data.csv"
    dataset = load_dataset("csv", data_files={"test": csv_path})
    test_data = dataset["test"]
    split_indices = torch.load(f"{base_path}/{dataset_name}/{model_name}_indices.pt", weights_only=False)

    return test_data, split_indices


def load_generated_tokens(dataset_name, model_name, split_indices, mode, base_path="results"):

    result_file = f"{base_path}/{model_name}/{dataset_name}/{dataset_name}_contri_results_manhattan.pkl"

    contri_chunks = load_pickle_file(result_file)
    contri_results = [x for chunk in contri_chunks for x in chunk]
    base_model_generated_tokens = [i[1]['generated tokens'] for i in contri_results]

    cleaned_generated_tokens = [x for i, x in enumerate(base_model_generated_tokens) if i not in split_indices['indices_to_remove']]
    remain_generated_tokens = [cleaned_generated_tokens[i] for i in split_indices['remain_idx']]
    type_generated_tokens =  [remain_generated_tokens[i] for i in split_indices[f'{mode}_idx']]
    num_type_generated_tokens = [len(tokens)+1 for tokens in type_generated_tokens]
    return num_type_generated_tokens


def create_prompt(question, dataset_name, selected_model):
    if dataset_name in ['nq','truthfulQA']:
        prompt = "Answer the following question using exactly one sentence.\nQuestion: " + question + "\nAnswer:"
    elif dataset_name in ['gsm8k'] and selected_model=='Llama-3-8B-Instruct':
        prompt = "Answer the following question using exactly a number. Do not explain.\nQuestion: " + question + "\nAnswer:"
    elif dataset_name in ['common'] and selected_model=='Gemma-3-4B-it':
        prompt = "Answer the following question by outputting exactly one of: A, B, C, D, or E. Do not explain.\nQuestion: " + question + "\nAnswer:"
    else:
        prompt = "Answer the following question using exactly one word. Do not explain.\nQuestion: " + question + "\nAnswer:"
    return prompt


def compute_focus_uncertainty(model, tokenizer, test_data, dataset_name,
                              num_generated_tokens=None,
                              model_path=None, selected_model=None):
    estimator = Focus(
        gamma=0.5,
        p=0.1,
        model_name=model_path,
        path=f"idf_cache/{selected_model.replace('/', '_')}_idf.pkl",
        idf_dataset="wikitext/wikitext-103-raw-v1", 
        trust_remote_code=False,
        idf_seed=42,
        idf_dataset_size=10000,  
        spacy_path="en_core_web_sm"
    )
    whitebox_model = WhiteboxModel(model, tokenizer)

    results = []

    for idx, example in enumerate(tqdm(test_data,
                                       file=sys.stderr,
                                       mininterval=0.5,
                                       ncols=100)):
        question = example["question"]
        prompt = create_prompt(question, dataset_name, selected_model)
        whitebox_model.generation_parameters = GenerationParameters(
            do_sample=False,
            max_new_tokens=num_generated_tokens[idx]
        )

        ue = estimate_uncertainty(whitebox_model, estimator, input_text=prompt)
        results.append(ue.uncertainty)

    return results


def save_results(results, output_path):
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(results))


def main():

    parser = argparse.ArgumentParser(
        description="Focus Baseline for Uncertainty Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            python Focus.py --dataset truthfulQA --model Gemma-3-4B-it --device cuda:0
            python Focus.py --dataset sciq --model Llama-3.2-3B-Instruct --device cuda:1
        """
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['sciq', 'simple_questions_wiki', 'nq', 'truthfulQA', 'gsm8k','common'],
        required=True,
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it','Llama-3-8B-Instruct'],
        required=True,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="results",
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default="results",
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default="processed_data",
    )

    
    args = parser.parse_args()


    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    for mode in ['train','val','test']:
        print(mode)

        test_data, split_indices = load_dataset_and_split(
            args.dataset,
            args.model,
            mode=mode,
            base_path=args.data_dir
        )

        num_generated_tokens = load_generated_tokens(
            args.dataset,
            args.model,
            split_indices=split_indices,
            mode=mode,
            base_path=args.input_dir
        )


        model_paths = {
            'Llama-3.2-3B-Instruct':  'meta-llama/Llama-3.2-3B-Instruct',
            'Gemma-3-4B-it': 'google/gemma-3-4b-it',
            'Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        }
        model_path = model_paths[args.model]

        uncertainty_scores = compute_focus_uncertainty(
            model,
            tokenizer,
            test_data,
            args.dataset,
            num_generated_tokens,
            model_path=model_path,
            selected_model=args.model
        )

        output_path = f"{args.output_dir}/{args.model}/{args.dataset}/{mode}/Focus.npy"
        save_results(uncertainty_scores, output_path)



if __name__ == "__main__":
    main()
