import argparse
import pathlib
import json
import os
import numpy as np
from scipy.spatial.distance import cosine

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

def cos_sim(mfcc_rsig: np.ndarray, mfcc_psig: np.ndarray) -> float:
    # Comparing the mean of each array is not very descriptive
    # Cosine similarity can be used to describe how similar the MFCCs of the two signals are
    # A higher similarity indicates that the timbre of the processed signal is closer to the original
    cos_sims = []
    for t in range(mfcc_rsig.shape[1]):
        frame_rsig = mfcc_rsig[:, t]
        frame_psig = mfcc_psig[:, t]
        
        # Subtract from 1 because this gives us distance, not similarity
        cos_sim = 1 - cosine(frame_rsig, frame_psig)
        cos_sims.append(cos_sim)

    # Convert list to numpy arrays
    cos_sims = np.array(cos_sims)

    # No normalization step here
    # Unlike other metrics, there isn't really a concept of better or worse timbre
    # Rather, we're interested in determining if the processed signal has a similar timbre to the original
    # Scores closer to 1 indicate high similarity, while scores closer to 0 indicate low similarity
    # We will still average the scores to get a single value
    avg_sim = np.mean(cos_sims, dtype=np.float64)

    return avg_sim

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


def parse_analysis_results_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze results from Samplifi run score.')
    parser.add_argument('--file', required=True, type=str, help='The analysis results file to parse')

    args = parser.parse_args()

    filepath = pathlib.Path(args.file)
    # Prepare output folder
    work_folder = pathlib.Path('./graphs')
    os.makedirs(work_folder, exist_ok=True)
    parse_analysis_results_file(filepath)