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

Usually against a dataset you will want scores. There are two scoring metrics: HAAQI, and spectral measures (from librosa, corresponding to musical perceptual qualities).

```
python run_samplifi.py --dataset medley_solos_db --score-haaqi --score-spectral
```

### Full usage

```
usage: run-samplifi.py [-h] [--input INPUT] [--output] [--dataset DATASET] [--sample-size SAMPLE_SIZE] [--score-haaqi] [--score-spectral] [--spectrogram] [--ddsp DDSP]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input
  --output              Write output files (this always happens when running against a single input file)
  --dataset DATASET     Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)
  --sample-size SAMPLE_SIZE
                        Number of samples to run against the dataset (0 for all samples)
  --score-haaqi         Compute HAAQI scores
  --score-spectral      Compute spectral evaluations of signal
  --spectrogram         Display spectrograms
  --ddsp DDSP           What instrument to attempt timbre transfer
```

## Downloading datasets

`run_samplifi.py` does not download any datasets and will fail if you haven't done so already. Download can be done with download_dataset.py:

```
python download_dataset --dataset medley_solos_db
```

## Using as a module

You can include `samplifi.py` as a module for your work. 

```
git submodule add git@github.com:hyve9/samplifi.git
git submodule update --init
```

Then in your code:

```
samplifi_dir = (Path.cwd() / 'samplifi/')
sys.path.append(str(samplifi_dir))
from samplifi import apply_samplifi

<code...>

# Load audio
orig_sarr, orig_sr = librosa.load(input_path, sr=None) # ndarray of amplitude values

# Run samplifi
sarr, marr, f0_contour, f0_mix, sr = apply_samplifi(orig_sarr, orig_sr)

```

I'll get around to making a proper PyPi module one day ;)


