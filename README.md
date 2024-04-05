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
python run-samplifi.py --input <input.wav>
```

Dataset:

```
python run-samplifi.py --dataset medley_solos_db
```

Usually against a dataset you will want scores. There are two scoring metrics: HAAQI, and spectral measures (from librosa, corresponding to musical perceptual qualities).

```
python run-samplifi.py --dataset medley_solos_db --score-haaqi --score-spectral
```

## Downloading datasets

`run-samplifi.py` does not download any datasets and will fail if you haven't done so already. Download can be done with download-mir-dataset.py
```
python download-mir-dataset.py --dataset medley_solos_db
```

## Running analysis

`analyze-results.py` can be used to analyze the results of a run, generating graphs and other statistical data. 

```
python analyze-results.py --file evaluation_medley_solos_db.json
```

### Full usage

```
usage: download-mir-dataset.py [-h] --dataset DATASET

Download an MIR dataset.

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  The name of the MIR dataset to download
```

```
usage: run-samplifi.py [-h] [--input INPUT] [--output] [--dataset DATASET] [--sample-size SAMPLE_SIZE] [--eval-haaqi] [--eval-spectral] [--titrate] [--spectrogram] [--audiogram] [--ddsp DDSP]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Input
  --output              Write output files (this always happens when running against a single input file)
  --dataset DATASET     Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)
  --sample-size SAMPLE_SIZE
                        Number of samples to run against the dataset (0 for all samples)
  --eval-haaqi          Compute HAAQI scores
  --eval-spectral       Compute spectral evaluations of signal
  --titrate             Try several different mixture ratios
  --spectrogram         Display spectrograms
  --audiogram           Display audiograms
  --ddsp DDSP           What instrument to attempt timbre transfer
```

```
usage: analyze-results.py [-h] --file FILE [--include-f0]

Analyze results from Samplifi run.

optional arguments:
  -h, --help    show this help message and exit
  --file FILE   The analysis results file to parse
  --include-f0  Include the f0 ratio data for line graphs
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


