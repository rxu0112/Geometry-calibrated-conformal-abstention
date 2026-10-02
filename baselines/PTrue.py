import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import *
from lm_polygraph.utils import estimate_uncertainty
import argparse
import pickle
from lm_polygraph.utils.generation_parameters import GenerationParameters
from tqdm import tqdm
from transformers import GenerationConfig

parser = argparse.ArgumentParser(description="PTrue Baseline")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument('--dataset', type=str, choices=['sciq', 'simple_questions_wiki', 'nq', 'truthfulQA','gsm8k','common'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it','Llama-3-8B-Instruct'], required=True)
args = parser.parse_args()

def load_pickle_file(path): 
    objs = [] 
    with open(path, "rb") as f: 
        while True: 
            try: 
                objs.append(pickle.load(f)) 
            except EOFError: 
                break
    return objs


# load base model
MODEL_IDS = {
    'Llama-3.2-3B-Instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    'Llama-3-8B-Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'Gemma-3-4B-it': 'google/gemma-3-4b-it',
}

if args.model in ['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct']:
    model_path = MODEL_IDS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 128001

    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, device_map=args.device)
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

if args.model == 'Gemma-3-4B-it':
    model_path = MODEL_IDS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, attn_implementation="eager", device_map=args.device)

base_model.resize_token_embeddings(len(tokenizer))
base_model.eval()

for type in ['train','val','test']:
    # load dataset
    data_path = f"processed_data/{args.dataset}/{args.model}_{type}_data.csv"
    data = load_dataset("csv", data_files={"test": data_path})["test"]

    # load result from proposed method for fairness
    split_indices = torch.load(f"processed_data/{args.dataset}/{args.model}_indices.pt", weights_only=False)
    contri_chunks = load_pickle_file(f"results/{args.model}/{args.dataset}/{args.dataset}_contri_results_manhattan.pkl")
    contri_results = [x for chunk in contri_chunks for x in chunk]
    base_model_generated_tokens = [i[1]['generated tokens'] for i in contri_results]

    cleaned_generated_tokens = [x for i, x in enumerate(base_model_generated_tokens) if i not in split_indices['indices_to_remove']]
    remain_generated_tokens = [cleaned_generated_tokens[i] for i in split_indices['remain_idx']]
    type_generated_tokens =  [remain_generated_tokens[i] for i in split_indices[f'{type}_idx']]
    num_type_generated_tokens = [len(tokens)+1 for tokens in type_generated_tokens]

    estimator = PTrue()
    model = WhiteboxModel(base_model, tokenizer)

    result = []
    for idx, example in enumerate(tqdm(data)):
        
        question = example["question"]

        if args.dataset in ['nq','truthfulQA']:
            prompt = "Answer the following question using exactly one sentence.\nQuestion: " + question + "\nAnswer:"
        elif args.dataset in ['gsm8k'] and args.model=='Llama-3-8B-Instruct':
            prompt = "Answer the following question using exactly a number. Do not explain.\nQuestion: " + question + "\nAnswer:"
        elif args.dataset in ['common'] and args.model=='Gemma-3-4B-it':
            prompt = "Answer the following question by outputting exactly one of: A, B, C, D, or E. Do not explain.\nQuestion: " + question + "\nAnswer:"
        else:
            prompt = "Answer the following question using exactly one word. Do not explain.\nQuestion: " + question + "\nAnswer:"

        model.generation_parameters = GenerationParameters(
            do_sample=False,
            max_new_tokens=num_type_generated_tokens[idx])
        
        ue = estimate_uncertainty(model, estimator, input_text=prompt)
        result.append(ue.uncertainty)

    np.save(f"results/{args.model}/{args.dataset}/{type}/PTrue.npy", np.array(result))