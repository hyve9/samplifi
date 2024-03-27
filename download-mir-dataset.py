import argparse
import mirdata

def download_dataset(dataset):
    data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
    data.download(force_overwrite=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download an MIR dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='The name of the MIR dataset to download')

    args = parser.parse_args()

    download_dataset(args.dataset)