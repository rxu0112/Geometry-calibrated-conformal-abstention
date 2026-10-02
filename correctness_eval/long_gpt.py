from openai import OpenAI
import argparse
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from general_utils import *

parser = argparse.ArgumentParser(description="Correctness determination using GPT")
parser.add_argument('--dataset', type=str, choices=['nq', 'truthfulQA'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Llama-3-8B-Instruct', 'Gemma-3-4B-it'], required=True)

args = parser.parse_args()

client = OpenAI(
    api_key="...",
    base_url="..."
)

    
def judge(judgement_model, question, generated_answer, hint):

    prompt = f"""
You are an impartial judge evaluating whether an answer correctly addresses a question.

Instructions:
- Output exactly ONE of the following tokens: "right", "wrong", or "I do not know".
- Do NOT provide explanations, reasoning, or any extra text.
- The hint is a reference answer but may not be the only correct answer.
- If the answer is clearly correct, output "right".
- If the answer is clearly incorrect, output "wrong".
- If correctness cannot be confidently determined, output "I do not know".

Question:
{question}

Hint:
{hint}

Answer:
{generated_answer}
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

data_path = f"processed_data/{args.dataset}/prepared.json"
dataset = load_dataset("json", data_files={"test": data_path})
eval_data = dataset["test"]

print(len(prediction))
print(len(eval_data))

try:
    with open(save_path, "r") as f:
        results = json.load(f)
    begin = len(results)
    eval_data = eval_data.select(range(begin, len(eval_data)))
    prediction = prediction[begin:]
    print(f"Loaded {begin} previous results from {save_path}")
except FileNotFoundError:
    results = []
    print("No previous results found. Starting fresh.")



for idx, sample in enumerate(tqdm(prediction)):
    
    question = eval_data[idx]["question"]
    hint  = eval_data[idx]['correct_answer']
    generated_answer = ''.join(sample[1]['generated tokens'])  # join tokens into string

    # Judge
    judgement = judge(judgement_model, question, generated_answer, hint)

    results.append(judgement)

    if (idx + 1) % 5 == 0:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

with open(save_path, "w") as f:
    json.dump(results, f, indent=2)