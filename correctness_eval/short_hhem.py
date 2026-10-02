from transformers import AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import argparse
from general_utils import *

parser = argparse.ArgumentParser(description="Correctness determination using HHEM")
parser.add_argument('--dataset', type=str, choices=['sciq', 'simple_questions_wiki','gsm8k','common'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Gemma-3-4B-it', 'Llama-3-8B-Instruct'], required=True)
args = parser.parse_args()

load_path = f"results/{args.model}/{args.dataset}/{args.dataset}_contri_results_manhattan.pkl"
save_path = f"results/{args.model}/{args.dataset}/{args.dataset}_hhem_judge.pt"


contri_chunks = load_pickle_file(load_path)
prediction = [x for chunk in contri_chunks for x in chunk]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)
model.eval()
model.to(device)

scores = []
for sample in tqdm(prediction):
    merged_true_answers = sample[1]['merged_true_answer']
    merged_prediction = sample[1]['merged_prediction']
    hem_score = model.predict([(merged_true_answers, merged_prediction)])[0]
    scores.append(hem_score.cpu().detach().numpy())

right_threshold = 0.8
wrong_threshold = 0.2
print(len([i for i in scores if i>right_threshold]))
print(len([i for i in scores if i<wrong_threshold]))

torch.save(scores,save_path)