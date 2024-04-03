import argparse
import pathlib
import json
import os

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