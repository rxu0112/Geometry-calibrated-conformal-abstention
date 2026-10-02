import argparse
import json
import os
import torch
from llama_representation.llama import LlamaAttributor, AttrConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import heapq
import numpy as np
from llama_representation.llama_utils import *
from general_utils import *

parser = argparse.ArgumentParser(
    description="Compute contributions using LlamaAttributor."
)
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
parser.add_argument("--i_block", type=int, default=100, help="take smaller value if memory is not enough")
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument('--dataset', type=str, choices=['sciq', 'simple_questions_wiki',
                                                    'nq', 'truthfulQA', 'gsm8k','common'], required=True, help="Dataset to use.")
parser.add_argument('--metric', type=str, default='manhattan', choices=['manhattan', 'dot'])
args = parser.parse_args()

if args.dataset in ['nq','truthfulQA']:
    word_limit = 50
else:
    word_limit = 10

if args.dataset in ["gsm8k"]:
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>"}
    stop_substrings = {"\n"}
elif args.dataset in ['common']:
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>", "."}
    stop_substrings = {"\n"}
else:
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>", "."}
    stop_substrings = set()

def generate_statement(question, answer):
    statement_prompt = (
        f"Convert the following Q&A into a single factual sentence.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Statement:"
    )

    encoding = merge_tokenizer(statement_prompt, return_tensors="pt").to(args.device)
    outputs = merge_model.generate(
        **encoding,
        do_sample=False,
        max_new_tokens=128,
        pad_token_id=merge_tokenizer.pad_token_id,
        eos_token_id=merge_tokenizer.eos_token_id,
    )
    statement = merge_tokenizer.decode(outputs[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return statement

merge_model_path = "Qwen/Qwen2.5-7B"
merge_model = AutoModelForCausalLM.from_pretrained(merge_model_path, dtype=torch.float16)
merge_tokenizer = AutoTokenizer.from_pretrained(merge_model_path)
if merge_tokenizer.pad_token is None:
    merge_tokenizer.pad_token = merge_tokenizer.eos_token
    merge_model.resize_token_embeddings(len(merge_tokenizer))
merge_model.eval()
merge_model.to(args.device)


model_path = "meta-llama/Llama-3.2-3B-Instruct"
cfg = AttrConfig(
    model_path=model_path,
    device=args.device,
    dtype=args.dtype,
    i_block=args.i_block,
)
tool = LlamaAttributor(cfg)

data_path = f"processed_data/{args.dataset}/prepared.json"
dataset = load_dataset("json", data_files={"test": data_path})
print(len(dataset['test']))

if args.dataset == 'nq':
    num_sample = 5000
    if len(dataset['test'])>num_sample:
        dataset["test"] = dataset["test"].select(range(num_sample))

eval_data = dataset["test"]
save_path = f"results/Llama-3.2-3B-Instruct/{args.dataset}/{args.dataset}_contri_results_{args.metric}.pkl" 

try:
    chunks = load_pickle_file(save_path)
    results = [x for chunk in chunks for x in chunk]
    begin = len(results)
    eval_data = eval_data.select(range(begin, len(eval_data)))
    print(f"Loaded {begin} previous results from {save_path}")
except FileNotFoundError:
    print("No previous results found. Starting fresh.")

chunk = []
with torch.no_grad():
    for idx, example in enumerate(tqdm(eval_data)):
        output = []
        question = example["question"]

        if args.dataset in ['nq','truthfulQA']:
            prompt = "Answer the following question using exactly one sentence.\nQuestion: " + question + "\nAnswer:"
        else:
            prompt = "Answer the following question using exactly one word. Do not explain.\nQuestion: " + question + "\nAnswer:"
        
        init_enc = tool.tokenizer(prompt, return_tensors="pt")
        init_enc = {k: v.to(args.device) for k, v in init_enc.items()}
        init_input_ids = init_enc["input_ids"]  # [1, T]
        init_tokens = [tool.tokenizer.decode([tid], skip_special_tokens=False)
                    for tid in init_input_ids[0].tolist()]
        output.append(init_tokens)
        
        tool.cfg.metric = args.metric
        concate_att_mlp_with_merged_residual(tool=tool, device=args.device, prompt=prompt, 
                                             stop_tokens=stop_tokens, stop_substrings=stop_substrings, word_limit=word_limit, output=output)

        prediction = ''.join(output[1]['generated tokens'])

        if args.dataset not in ['nq','truthfulQA']:
            merged_prediction = generate_statement(question, prediction)
            merged_true_answer = generate_statement(question, example["correct_answer"])
            output[1]['merged_prediction'] = merged_prediction
            output[1]['merged_true_answer'] = merged_true_answer

        chunk.append(output)

        if (idx + 1) % 50 == 0:
            append_pickle(chunk, save_path)
            chunk.clear()

if len(chunk) > 0:
    append_pickle(chunk, save_path)
    chunk.clear()