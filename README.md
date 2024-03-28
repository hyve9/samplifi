# Samplifi

Add harmonic content to music to help hearing impaired listeners hear music better.

### Requirements

Install packages:

```
sudo apt install libasound2-dev portaudio19-dev
```

Create conda environment:

```
conda create -f environment.yml -n samplifi
conda activate samplifi
```

## Running

Samplifi can be run on single inputs or MIR datasets. Single input

```
python run_samplifi.py --input <input.wav>
```

Dataset:

```
python run_samplifi.py --dataset medley_solos_db
```

Usually against a dataset you will want scores. There are three scoring metrics: HAAQI, spectral measures (from librosa), and experimental musical measures (corresponding to musical perceptual qualities).

```
python run_samplifi.py --dataset medley_solos_db --score-haaqi --score-spectral --score-musical
```

### Full usage

```
usage: run-samplifi.py [-h] [--input INPUT] [--output] [--dataset DATASET] [--sample-size SAMPLE_SIZE] [--score-haaqi] [--score-spectral] [--score-musical] [--spectrogram] [--ddsp DDSP]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input
  --output              Write output files (this always happens when running against a single input file)
  --dataset DATASET     Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)
  --sample-size SAMPLE_SIZE
                        Number of samples to run against the dataset (0 for all samples)
  --score-haaqi         Compute HAAQI scores
  --score-spectral      Compute spectral evaluations of signal
  --score-musical       Compute musical evaluations of signal
  --spectrogram         Display spectrograms
  --ddsp DDSP           What instrument to attempt timbre transfer
```

## Downloading datasets

Can be done with download_dataset.py:

```
python download_dataset --dataset medley_solos_db
```
