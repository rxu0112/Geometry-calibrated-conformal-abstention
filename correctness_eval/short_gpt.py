from openai import OpenAI
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from general_utils import *

parser = argparse.ArgumentParser(description="Correctness determination using GPT")
parser.add_argument('--dataset', type=str, choices=['sciq', 'gsm8k', 'simple_questions_wiki'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct'], required=True)
args = parser.parse_args()

client = OpenAI(
    api_key="...",
    base_url="..."
)

    
def judge(judgement_model, question, generated_answer):

    prompt = f"""
Judge if the answer is correct to the question. 

Rules:
1. Only output exactly one of the three responses: "right", "wrong", or "I do not know".
2. Do NOT output explanations.

Question: {question}

Answer: {generated_answer}
    """

    res = client.chat.completions.create(
        model=judgement_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()

judgement_model = "gpt-5.1"  # or "gpt-4o"
print(judgement_model)
save_path = f"results/{args.model}/{args.dataset}/{args.dataset}_{judgement_model}_judge.json"
contri_chunks = load_pickle_file(f"results/{args.model}/{args.dataset}/{args.dataset}_contri_results_manhattan.pkl")
prediction = [x for chunk in contri_chunks for x in chunk]
hem_examples = torch.load(f"results/{args.model}/{args.dataset}/{args.dataset}_hhem_judge.pt", weights_only=False)

data_path = f"processed_data/{args.dataset}/prepared.json"
dataset = load_dataset("json", data_files={"test": data_path})
eval_data = dataset["test"]


eval_data = [v for i, v in enumerate(eval_data) if hem_examples[i] < 0.2]
prediction = [v for i, v in enumerate(prediction) if hem_examples[i] < 0.2]

print(len(prediction))
print(len(eval_data))

try:
    with open(save_path, "r") as f:
        results = json.load(f)
    begin = len(results)
    eval_data = eval_data[begin:]
    prediction = prediction[begin:]
    print(f"Loaded {begin} previous results from {save_path}")
except FileNotFoundError:
    results = []
    print("No previous results found. Starting fresh.")



for idx, sample in enumerate(tqdm(prediction)):
    
    question = eval_data[idx]["question"]
    generated_answer = ''.join(sample[1]['generated tokens'])  # join tokens into string

    # Judge
    judgement = judge(judgement_model, question, generated_answer)

    results.append(judgement)

    if (idx + 1) % 10 == 0:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

with open(save_path, "w") as f:
    json.dump(results, f, indent=2)