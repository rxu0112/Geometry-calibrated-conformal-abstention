import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import torch.nn.functional as F
import pandas as pd
from general_utils import *

parser = argparse.ArgumentParser(
    description="Compute logit."
)
parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument('--dataset', type=str, choices=['sciq', 'simple_questions_wiki',
                                                    'nq', 'truthfulQA', 'gsm8k', 'common'], required=True)
args = parser.parse_args()

def _dtype_from_str(s: str):
    if s == "fp16": return torch.float16
    if s == "bf16": return torch.bfloat16
    return torch.float32

if args.dataset in ['nq','truthfulQA']:
    word_limit = 50
else:
    word_limit = 10

if args.dataset == "gsm8k":
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>"}
    stop_substrings = {"\n"}
elif args.dataset in ['simple_questions_wiki','sciq','common']:
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>","."}
    stop_substrings = {"\n"}
else:
    stop_tokens = {"<|eot_id|>", "<|endoftext|>", "</s>", "."}
    stop_substrings = set()

lens_path = f'results/Llama-3-8B-Instruct/{args.dataset}/{args.dataset}_lens_results.pkl'
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=_dtype_from_str(args.dtype))
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
model.to(args.device)
model.eval()

# print(model)

NB_LAYERS = len(model.model.layers)  # = 28

def hook_output(layer_id):
    def forward_hook(module, input, output):
        # output: (batch, seq_len, hidden_dim)
        output_saving.append(output[0])
    return forward_hook

def hook_input(layer_id):
    def hook(module, input, output):
        # shape: (batch, seq_len, hidden_dim)
        output_saving.append(input[0])
    return hook


# Register hooks
for k in range(NB_LAYERS):
    model.model.layers[k].post_attention_layernorm.register_forward_hook(hook_input(k))
    model.model.layers[k].register_forward_hook(hook_output(k))

data_path = f"processed_data/{args.dataset}/prepared.json"
dataset = load_dataset("json", data_files={"test": data_path})

if args.dataset == 'nq':
    num_sample = 5000
    if len(dataset['test'])>num_sample:
        dataset["test"] = dataset["test"].select(range(num_sample))

eval_data = dataset["test"]

try:
    chunks = load_pickle_file(lens_path)
    prob_history_examples = sum([chunk['prob_history_examples'] for chunk in chunks], [])
    begin = len(prob_history_examples)
    eval_data = eval_data.select(range(begin, len(eval_data)))
    print(f"Loaded {begin} previous results from {lens_path}")
except FileNotFoundError:
    print("No previous results found. Starting fresh.")

logit_history_examples = []
prob_history_examples = []
h_last_history_examples = []
generated_tokens_examples = []


for idx, example in enumerate(tqdm(eval_data)):
    question = example["question"]

    if args.dataset in ['nq','truthfulQA']:
        prompt = "Answer the following question using exactly one sentence.\nQuestion: " + question + "\nAnswer:"
    elif args.dataset in ['gsm8k']:
        prompt = "Answer the following question using exactly a number. Do not explain.\nQuestion: " + question + "\nAnswer:"
    else:
        prompt = "Answer the following question using exactly one word. Do not explain.\nQuestion: " + question + "\nAnswer:"
    
    logit_history_tokens = []
    prob_history_tokens = []
    h_last_history_tokens = []
    generated_tokens = []
    for w in range(word_limit):
        output_saving = []        
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        with torch.no_grad():
            _ = model(**inputs)
        
        W_E = model.lm_head.weight  # (vocab_size, hidden_size)
        vocab = tokenizer.get_vocab()
        last_token_idx = -1

        for layer_id, h in enumerate(output_saving):
            if h.dim() == 2:  # (seq, hidden)
                output_saving[layer_id] = h.unsqueeze(0) # ensure (batch, seq_len, hidden)

        final_h = model.model.norm(output_saving[-1])[0, last_token_idx, :] # output_saving is in (layer, batch, seq_len, hidden_dim)
        final_logits = torch.matmul(final_h, W_E.T)
        final_top1_id = torch.argmax(final_logits).item()
        next_token = tokenizer.decode([final_top1_id], skip_special_tokens=False)

        if tokenizer.eos_token_id is not None and final_top1_id == tokenizer.eos_token_id:
            break
        if next_token.strip() in stop_tokens or any(s in next_token for s in stop_substrings):
            break
        
        logit_history = []
        prob_history = []
        h_last_history = []
        for layer_id, h in enumerate(output_saving):
            # h: (1, seq_len, hidden_size)
            h_last_normalized = model.model.norm(h)[0, last_token_idx, :]  # hidden of last token, a normalization is applied before unembedding

            # Project to vocab space
            logits = torch.matmul(h_last_normalized, W_E.T)
            probs = F.softmax(logits.float(), dim=-1).detach().cpu().numpy()
            inter_top1_logit = logits[final_top1_id].item()
            inter_top1_prob = probs[final_top1_id].item()

            logit_history.append(inter_top1_logit)
            prob_history.append(inter_top1_prob)
            h_last_history.append(h_last_normalized.detach().cpu().numpy())


        logit_history_tokens.append(logit_history)
        prob_history_tokens.append(prob_history)
        h_last_history_tokens.append(h_last_history)
        generated_tokens.append(next_token)
        prompt += next_token

    logit_history_examples.append(logit_history_tokens)
    prob_history_examples.append(prob_history_tokens)
    h_last_history_examples.append(h_last_history_tokens)
    generated_tokens_examples.append(generated_tokens)
        
    if (idx + 1) % 50 == 0:
        lens_results_chunk = {
            'logit_history_examples': logit_history_examples,
            'prob_history_examples': prob_history_examples,
            'h_last_history_examples': h_last_history_examples,
            'generated_tokens_examples': generated_tokens_examples,
        }

        append_pickle(lens_results_chunk, lens_path)

        # VERY IMPORTANT: clear memory
        logit_history_examples.clear()
        prob_history_examples.clear()
        h_last_history_examples.clear()
        generated_tokens_examples.clear()

if len(logit_history_examples) > 0:
    lens_results_chunk = {
        'logit_history_examples': logit_history_examples,
        'prob_history_examples': prob_history_examples,
        'h_last_history_examples': h_last_history_examples,
        'generated_tokens_examples': generated_tokens_examples,
    }
    append_pickle(lens_results_chunk, lens_path)
