import argparse
import mirdata
import logging

def download_dataset(dataset, silent=False):
    # Save the original logging level
    original_level = logging.getLogger().getEffectiveLevel()

    if silent:
        # Set logging level to WARNING
        logging.basicConfig(level=logging.WARNING)

    data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
    data.download(force_overwrite=False)

    if silent:
        # Restore the original logging level
        logging.getLogger().setLevel(original_level)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download an MIR dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='The name of the MIR dataset to download')
    parser.add_argument('--silent', action='store_true', help='Suppress stdout output')

    args = parser.parse_args()

    download_dataset(args.dataset, args.silent)