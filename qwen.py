import json
import os
from typing import List, Dict
import re

import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from Reasoning_Flow.utils_stat import (
    pairwise_similarity,
    pairwise_menger_curvature_similarity
)


def parse_proof_chain_to_steps(proof_chain):
    """Parse proof chain into individual step lines"""
    if not isinstance(proof_chain, str):
        return []
    
    lines = [line.strip() for line in proof_chain.split('\n') if line.strip()]
    steps = []
    
    for line in lines:
        # Skip header lines
        if line.startswith('Proof chain:') or line == 'Proof chain':
            continue
        
        # Clean and add non-empty lines
        clean_line = line.strip()
        if clean_line:
            steps.append(clean_line)
    
    return steps


@torch.no_grad()
def extract_reasoning_trajectory(
    cot_steps: List[str],
    tokenizer,
    model,
    device: str = "mps"
) -> np.ndarray:
    """
    Extract reasoning trajectory from CoT steps using Qwen (decoder-only model).
    Uses last hidden layer and mean-pooling.
    """
    trajectories = []
    context = ""

    for step in cot_steps:
        # Build cumulative context
        context = context + " " + step if context else step

        # Tokenize
        encoding = tokenizer(
            context,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Forward pass to get hidden states
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Last hidden layer
        last_hidden = outputs.hidden_states[-1].squeeze(0)  # [seq, dim]

        # Mean pool (ignore padding)
        mask = attention_mask.squeeze(0).bool()
        step_vec = last_hidden[mask].mean(dim=0).cpu().numpy()

        trajectories.append(step_vec)

    return np.stack(trajectories).astype(np.float32)


def extract_trajectories_from_dataframe(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str = "mps",
) -> Dict[str, np.ndarray]:
    """
    Extract reasoning trajectories for all examples in dataframe.
    
    Returns:
        {example_id: trajectory_array, ...}
        Each trajectory is [num_steps, hidden_dim]
    """
    model.eval()
    trajectories = {}
    
    print(f"\nExtracting trajectories for {len(df)} examples")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Parse proof chain into steps (just split by lines)
        cot_steps = parse_proof_chain_to_steps(row['Complex_CoT'])
        
        if idx == df.index[0]:  # Print first example
            print(f"\nExample {idx} CoT steps:")
            for step in cot_steps[:3]:  # Print first 3 steps
                print(f"  {step}")
            if len(cot_steps) > 3:
                print(f"  ... and {len(cot_steps) - 3} more steps")
            print(f"Total steps: {len(cot_steps)}")
        
        # Extract trajectory
        traj = extract_reasoning_trajectory(cot_steps, tokenizer, model, device)
        
        if idx == df.index[0]:  # Print first trajectory info
            print(f"Trajectory shape: {traj.shape}")
            print(f"First step vector (first 5 dims): {traj[0, :5]}")
        
        # Label by index
        label = f"ex_{idx}"
        trajectories[label] = traj
    
    return trajectories


def compute_similarities(
    trajectories: Dict[str, np.ndarray],
    output_dir: str
) -> pd.DataFrame:
    """
    Compute curvature similarities between trajectories.
    """
    if len(trajectories) < 2:
        print(f"Need at least 2 examples, got {len(trajectories)}")
        return pd.DataFrame()
    
    print(f"\nComputing curvature similarity for {len(trajectories)} examples")
    
    # Debug: print trajectory shapes
    print("\nTrajectory shapes:")
    for label, traj in list(trajectories.items())[:3]:
        print(f"  {label}: {traj.shape}")
    
    # Convert to list format
    traj_list = list(trajectories.values())
    labels = list(trajectories.keys())
    
    # Menger curvature similarity (Pearson correlation)
    _, curv_sim = pairwise_menger_curvature_similarity(
        traj_list,
        metric='pearson',
        align='truncate'
    )
    
    print(f"\nCurvature similarity matrix shape: {curv_sim.shape}")
    print(f"Sample values from matrix:\n{curv_sim[:3, :3]}")
    
    # Extract upper triangle (excluding diagonal) for statistics
    n = len(traj_list)
    upper_tri = np.triu_indices(n, k=1)
    curv_vals = curv_sim[upper_tri]
    
    print(f"\nNumber of pairwise similarities: {len(curv_vals)}")
    print(f"Non-zero similarities: {np.count_nonzero(curv_vals)}")
    
    result = {
        'Num_Examples': len(trajectories),
        'Num_Pairs': len(curv_vals),
        'Curvature_Mean': float(np.mean(curv_vals)),
        'Curvature_Std': float(np.std(curv_vals)),
        'Curvature_Min': float(np.min(curv_vals)),
        'Curvature_Max': float(np.max(curv_vals)),
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw matrix as CSV
    pd.DataFrame(curv_sim, index=labels, columns=labels).to_csv(
        os.path.join(output_dir, "curvature_similarity.csv")
    )
    
    print(f"  Curvature: {result['Curvature_Mean']:.4f} ± {result['Curvature_Std']:.4f}")
    print(f"  Range: [{result['Curvature_Min']:.4f}, {result['Curvature_Max']:.4f}]")
    
    return pd.DataFrame([result])


def main():
    # Configuration
    device = "mps"
    parquet_file = "/Users/arkarutvik/Downloads/deepseek_2k_RP_test.parquet"
    output_dir = "./trajectory_results_depthwise"
    
    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("causality-grammar/qwen3-1.7B-fullfinetuned")
    model = AutoModelForCausalLM.from_pretrained(
        "causality-grammar/qwen3-1.7B-fullfinetuned",
        output_hidden_states=True
    )
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Depth range
    depth_min = 4
    depth_max = 11
    
    all_results = []

    for depth in range(depth_min, depth_max + 1):
        print("\n" + "="*60)
        print(f"PROCESSING DEPTH = {depth}")
        print("="*60)

        df_depth = df[df['Depth'] == depth]  # take first 10 each depth

        print(f"✓ Filtered to {len(df_depth)} examples with Depth={depth}")
        if len(df_depth) < 2:
            print("Skipping — need at least 2 examples")
            continue

        # Extract reasoning trajectories
        print("\nEXTRACTING REASONING TRAJECTORIES")
        trajectories = extract_trajectories_from_dataframe(
            df_depth,
            tokenizer,
            model,
            device
        )

        # Compute similarities
        print("\nCOMPUTING SIMILARITIES")
        depth_dir = os.path.join(output_dir, f"depth_{depth}")
        results_df = compute_similarities(trajectories, depth_dir)

        # Add depth column to results
        if not results_df.empty:
            results_df.insert(0, "Depth", depth)
            all_results.append(results_df)

    # Merge all depth results
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)

        summary.to_csv(os.path.join(output_dir, "depthwise_summary.csv"), index=False)

        with open(os.path.join(output_dir, "depthwise_summary.json"), "w") as f:
            json.dump(summary.to_dict(orient="records"), f, indent=2)

        print("\n" + "="*60)
        print("✓ ALL DEPTH RESULTS SAVED")
        print("="*60)
        print(f"Saved to: {os.path.abspath(output_dir)}")
        print(" - depthwise_summary.csv")
        print(" - depthwise_summary.json")
    else:
        print("\nNo valid depth results were computed.")


if __name__ == "__main__":
    main()