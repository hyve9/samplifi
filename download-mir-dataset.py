import os
import sys
import argparse
import mirdata

def download_dataset(dataset, silent=False):
    # Save the original stdout
    original_stdout = sys.stdout

    if silent:
        # Set stdout to null
        sys.stdout = open(os.devnull, 'w')

    try:
        data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
        data.download(force_overwrite=False)
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download an MIR dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='The name of the MIR dataset to download')
    parser.add_argument('--silent', action='store_true', help='Suppress stdout output')

    args = parser.parse_args()

    download_dataset(args.dataset, args.silent)