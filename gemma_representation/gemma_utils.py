import argparse
import json
import os
import torch
from gemma_representation.gemma import GemmaAttributor, AttrConfig
from datasets import load_dataset
from tqdm import tqdm
import heapq
import numpy as np
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import pickle

def concate_att_mlp_with_merged_residual(tool, device: str, prompt: str, word_limit: int = 10, output: list=[]):
    contrib_mats_per_step = []
    generated_tokens = []
    for w in range(word_limit):
        next_id, next_txt = tool.generate_one_token(
            prompt, do_sample=False, temperature=1.0, top_p=1.0
        )
        
        if tool.tokenizer.eos_token_id is not None and next_id == tool.tokenizer.eos_token_id:
            break
        if next_txt == "\n":
            break
        if next_txt.strip() in {"<|eot_id|>", "<|endoftext|>", "</s>", ".", "<end_of_turn>"}:
            break
        
        enc = tool.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_ids = enc["input_ids"]  # [1, T]
        batch_size, seq_len = input_ids.shape
        attention_mask = enc['attention_mask']
        
        oproj_refs, outputs_after_mlp, outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs = tool._capture_o_proj_refs(enc)
        att_contrib_mats, mlp_contrib_mats, mlp_res_mats = tool._concate_att_mlp_with_merged_residual(input_ids, oproj_refs, outputs_after_mlp, outputs_after_post_attention, outputs_after_pre_feedforward, outputs_after_gate_proj, outputs_after_up_proj, attn_probs_refs, v_refs, attention_mask)
        att_con_mats_np = []
        mlp_con_mats_np = []
        mlp_res_mats_np= []

        for mats in att_contrib_mats:
            att_con_mats_np.append(mats[-1:, :].detach().float().cpu().numpy().copy())
        for mats in mlp_contrib_mats:
            mlp_con_mats_np.append(mats.detach().float().cpu().numpy())
        for mats in mlp_res_mats:
            mlp_res_mats_np.append(mats.detach().float().cpu().numpy())
        
        contrib_mats_per_step.append([att_con_mats_np, mlp_con_mats_np, mlp_res_mats_np])
        
        generated_tokens.append(next_txt)

        print(next_txt)
        prompt+=next_txt

    output.append({"generated tokens": generated_tokens, 
                   "metric": tool.cfg.metric, 
                   "contrib mats loop": contrib_mats_per_step})