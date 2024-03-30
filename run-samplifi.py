import os
import random
import sys
# import numpy as np
from scipy.io import wavfile
import librosa
import pathlib
import argparse
import csv
from tensorflow.python.ops.numpy_ops import np_config
from contextlib import ExitStack
np_config.enable_numpy_behavior()

from clarity.utils.audiogram import (
    AUDIOGRAM_REF, 
    AUDIOGRAM_MILD, 
    AUDIOGRAM_MODERATE, 
    AUDIOGRAM_MODERATE_SEVERE
)

from samplifi import (
    eval_haaqi,
    eval_spectral,
    compute_timbre_transfer,
    plot_spectrogram,
    apply_samplifi,
)

test_ags = {'ref': AUDIOGRAM_REF, 
            'mild': AUDIOGRAM_MILD, 
            'moderate': AUDIOGRAM_MODERATE, 
            'severe': AUDIOGRAM_MODERATE_SEVERE}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Input')
    parser.add_argument('--output', action='store_true', help='Write output files (this always happens when running against a single input file)')
    parser.add_argument('--dataset', type=str, help='Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)')
    parser.add_argument('--sample-size', type=int, default=0, help='Number of samples to run against the dataset (0 for all samples)')
    parser.add_argument('--score-haaqi', action='store_true', help='Compute HAAQI scores')
    parser.add_argument('--score-spectral', action='store_true', help='Compute spectral evaluations of signal')
    parser.add_argument('--spectrogram', action='store_true', help='Display spectrograms')
    parser.add_argument('--ddsp', type=str, help='What instrument to attempt timbre transfer')

    args = parser.parse_args()

    if args.input and args.dataset:
        print('Cannot provide both --input and --dataset.')
        sys.exit(1)
    if args.input:
        input_path = pathlib.Path(args.input)
        write_output = True
        filename_suffix = '_single_input'
    elif args.dataset:
        dataset = args.dataset
        write_output = args.output
        filename_suffix = f'_{dataset}'
    else:
        print('Must provide either --input or --dataset.')
        sys.exit(1)
    score_haaqi = args.score_haaqi
    score_spectral = args.score_spectral
    spectrogram = args.spectrogram
    dataset = args.dataset
    sample_size = args.sample_size
    target_inst = args.ddsp if args.ddsp else False
    if target_inst:
        print('Timbre transfer is broken ;-;, exiting...)')
        sys.exit(1)
    
    # Create exit stack to handle file closing
    with ExitStack() as stack:
        # Prepare score files
        if score_haaqi:
            haaqi_file = open(f'haaqi_scores{filename_suffix}.csv', 'w', newline='')
            stack.enter_context(haaqi_file)
            haaqi_writer = csv.writer(haaqi_file)
            haaqi_writer.writerow(['Audiogram', 
                                   'Track_ID', 
                                   'Comparison', 
                                   'Score', 
                                   'Instrument', 
                                   'Genre', 
                                   'Drum'])
            scores = {'ref': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}

        if score_spectral:
            spectral_file = open(f'spectral_scores{filename_suffix}.csv', 'w', newline='')
            stack.enter_context(spectral_file)
            spectral_writer = csv.writer(spectral_file)
            spectral_writer.writerow(['Audiogram', 
                                      'Track_ID', 
                                      'Comparison', 
                                      'Pitch Detection',
                                      'Melodic Contour',
                                      'Timbre Preservation',
                                      'Harmonic Energy',
                                      'Instrument', 
                                      'Genre', 
                                      'Drum'])
            scores = {'ref': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}

        if dataset:
            import mirdata
            if dataset not in mirdata.list_datasets():
                print('Dataset not found')
                sys.exit(1)
            data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
            track_ids = random.sample(data.track_ids, sample_size) if sample_size else data.track_ids
            write_output = False
        elif input_path:
            # List with single track_id
            track_ids = [str(input_path.stem)]
        for i, track_id in enumerate(track_ids, start=1):
            print(f"Processing track {i} of {len(track_ids)}")
            # Hacky (hah) attempt to run the same code for mir datasets and single inputs
            # Prepare track and track metadata
            if args.input:
                track = None
                metadata = dict()
            else:
                track = data.track(track_id)
                input_path = pathlib.Path(track.audio_path)
                instrument = track.instrument if hasattr(track, 'instrument') else None
                genre = track.genre if hasattr(track, 'genre') else None
                drum = track.drum if hasattr(track, 'drum') else None
                # Why both a dict and a list? We may require the dictionary later
                metadata = {'instrument': instrument, 'genre': genre, 'drum': drum}
            metadata_values = list(metadata.values())

            print(f'Processing {input_path}...')

            # Load audio
            orig_sarr, orig_sr = librosa.load(input_path, sr=None) # ndarray of amplitude values
        
            # Run samplifi
            sarr, marr, f0_contour, f0_mix, sr = apply_samplifi(orig_sarr, orig_sr)

            # Try timbre transfer
            if target_inst:
                timbre_transfer = compute_timbre_transfer(f0_contour, target_inst)
                #Fix shape
                timbre_transfer = timbre_transfer.numpy()[0]
            else:
                timbre_transfer = None

            # Save spectrogram to file
            if spectrogram:
                # Three rows for original, f0, and mix; add a fourth row for timbre transfer if provided
                rows = 4 if target_inst else 3
                plot_spectrogram(track_id, rows, sarr, f0_contour, f0_mix, timbre_transfer)

            if write_output:
                # Prepare output folder
                work_folder = pathlib.Path('./output')
                os.makedirs(work_folder, exist_ok=True)
                # Prepare filenames
                filename_prefix = input_path.stem
                # Resample to original rate
                resamp_f0_mix = librosa.resample(f0_mix, orig_sr=sr, target_sr=orig_sr)
                resamp_f0_contour = librosa.resample(f0_contour, orig_sr=sr, target_sr=orig_sr)         

                # 6. Write out files
                marr.write(str(work_folder.joinpath(filename_prefix + '.mid')))
                wavfile.write(work_folder.joinpath(filename_prefix + '_f0.wav'), orig_sr, resamp_f0_contour)
                wavfile.write(work_folder.joinpath(filename_prefix + '_boosted.wav'), orig_sr, resamp_f0_mix)
                if target_inst:
                    resamp_timbre_transfer = librosa.resample(timbre_transfer, orig_sr=sr, target_sr=orig_sr)
                    wavfile.write(work_folder.joinpath(filename_prefix + '_timbre_transfer.wav'), orig_sr, resamp_timbre_transfer)

            if score_haaqi:
                for ag in test_ags:
                    scores[ag]['ref_v_f0'] = {'score': eval_haaqi(sarr, f0_contour, sr, sr, test_ags[ag]), **metadata}
                    scores[ag]['ref_v_mix'] = {'score': eval_haaqi(sarr, f0_mix, sr, sr, test_ags[ag]), **metadata}
                    for score in scores[ag]:
                        print(f'HAAQI evaluation score for {score} against audiogram_{ag}: {scores[ag][score]}')
                        haaqi_writer.writerow([ag, 
                                               track_id, 
                                               score, 
                                               scores[ag][score], 
                                               *metadata_values])

            if score_spectral:
                for ag in test_ags:
                    scores[ag]['ref_v_f0'] = {**eval_spectral(sarr, f0_contour, sr, sr, test_ags[ag]), **metadata}
                    scores[ag]['ref_v_mix'] = {**eval_spectral(sarr, f0_mix, sr, sr, test_ags[ag]), **metadata}
                    for score in scores[ag]:
                        print(f'Spectral evaluation score for {score} against audiogram_{ag}: {scores[ag][score]}')
                        spectral_writer.writerow([ag,
                                                  track_id, 
                                                  score, 
                                                  scores[ag][score]['pitch_detection'],
                                                  scores[ag][score]['melodic_contour'],
                                                  scores[ag][score]['timbre_preservation'],
                                                  scores[ag][score]['harmonic_energy'], 
                                                  *metadata_values])


        