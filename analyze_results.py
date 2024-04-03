import argparse
import pathlib
import json
import os
import numpy as np

def sigmoid(x):
    # Range between -1 and 1
    # Clip and scale x to prevent occasional RuntimeWarning: overflow encountered in exp
    x = np.clip(x, -512, 512)
    
    # Adjust scale based on the magnitude of x
    scale = 1.0
    if np.abs(x).max() > 100:
        scale = 0.01
    elif np.abs(x).max() > 10:
        scale = 0.1

    x = x * scale
    return 2 / (1 + np.exp(-x)) - 1


def get_score(ref: np.ndarray, proc: np.ndarray) -> float:
    """Get score between two evaluation metrics

    Args:
        a: original evaluation metric
        b: processed evaluation metric

    Returns:
        Score between two metrics
    """
    # Check lengths
    if len(ref) != len(proc):
        # Just diff the averages
        diff = np.mean(proc, dtype=np.float64) - np.mean(ref, dtype=np.float64)
    else:
        # Compute a difference between the two averages
        diff = proc - ref

    # Normalize between -1 and 1
    score = sigmoid(diff)

    # Average
    average = np.mean(score, dtype=np.float64)
    
    return average

def scores_table(data):
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze results from Samplifi run.')
    parser.add_argument('--file', required=True, type=str, help='The analysis results file to parse')

    args = parser.parse_args()

    filepath = pathlib.Path(args.file)
    # Prepare output folder
    work_folder = pathlib.Path('./graphs')
    os.makedirs(work_folder, exist_ok=True)
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(data)