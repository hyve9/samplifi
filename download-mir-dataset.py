import argparse
import mirdata

def download_dataset(dataset_name):
    data = mirdata.initialize(dataset_name, data_home=f'./mir_datasets/{dataset_name}')
    data.download(force_overwrite=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download an MIR dataset.')
    parser.add_argument('--dataset', required=True, type=str, help='The name of the MIR dataset to download')

    args = parser.parse_args()

    download_dataset(args.dataset)