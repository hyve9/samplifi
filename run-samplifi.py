import os
import random
import sys
from scipy.io import wavfile
import json
import librosa
import pathlib
import argparse
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from clarity.utils.audiogram import (
    AUDIOGRAM_REF, 
    AUDIOGRAM_MILD, 
    AUDIOGRAM_MODERATE, 
    AUDIOGRAM_MODERATE_SEVERE
)

from samplifi import (
    run_haaqi,
    get_spectral_features,
    compute_timbre_transfer,
    plot_spectrogram,
    apply_samplifi,
)

test_ags = {'normal': AUDIOGRAM_REF, 
            'mild': AUDIOGRAM_MILD, 
            'moderate': AUDIOGRAM_MODERATE, 
            'severe': AUDIOGRAM_MODERATE_SEVERE
            }

f0_ratios = {'0.25': 
                { 'value':0.25,
                  'f0_mix': None,
                 },
                '0.5': 
                { 'value':0.5,
                  'f0_mix': None,
                 },
                '0.75': 
                { 'value':0.75,
                  'f0_mix': None,
                 }
            }   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Input')
    parser.add_argument('--output', action='store_true', help='Write output files (this always happens when running against a single input file)')
    parser.add_argument('--dataset', type=str, help='Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)')
    parser.add_argument('--sample-size', type=int, default=0, help='Number of samples to run against the dataset (0 for all samples)')
    parser.add_argument('--eval-haaqi', action='store_true', help='Compute HAAQI scores')
    parser.add_argument('--eval-spectral', action='store_true', help='Compute spectral evaluations of signal')
    parser.add_argument('--titrate', action='store_true', help='Try several different mixture ratios')
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
    eval_haaqi = args.eval_haaqi
    eval_spectral = args.eval_spectral
    spectrogram = args.spectrogram
    dataset = args.dataset
    sample_size = args.sample_size
    titrate = args.titrate
    target_inst = args.ddsp if args.ddsp else False
    if target_inst:
        print('Timbre transfer is broken ;-;, exiting...)')
        sys.exit(1)

    tracks = dict()

    if dataset:
        import mirdata
        if dataset not in mirdata.list_datasets():
            print('Dataset not found')
            sys.exit(1)
        data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
        track_ids = random.sample(data.track_ids, sample_size) if sample_size else data.track_ids
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
            metadata = {'instrument': instrument, 'genre': genre, 'drum': drum}

        print(f'Processing {input_path}...')

        # Load audio
        orig_sarr, orig_sr = librosa.load(input_path, sr=None) # ndarray of amplitude values
    
        # Run samplifi
        sarr, marr, f0_contour, f0_mix, sr = apply_samplifi(orig_sarr, orig_sr)
        if titrate:
            # This is only for scoring, we don't use titration for spectrogram or output
            for f0_ratio in f0_ratios:
                f0_ratios[f0_ratio]['f0_mix'] = f0_contour * f0_ratios[f0_ratio]['value'] + sarr * (1 - f0_ratios[f0_ratio]['value'])

        # Try timbre transfer
        if target_inst:
            timbre_transfer = compute_timbre_transfer(f0_contour, target_inst)
            #Fix shape
            timbre_transfer = timbre_transfer.numpy()[0]
        else:
            timbre_transfer = None

        # Save spectrogram to file
        if spectrogram:
            if titrate:
                print('Not displaying spectrogram for titrated f0_mixes, using default ratio')
            # Three rows for original, f0, and mix; add a fourth row for timbre transfer if provided
            rows = 4 if target_inst else 3
            plot_spectrogram(track_id, rows, sarr, f0_contour, f0_mix, timbre_transfer)

        if write_output:
            # Prepare output folder
            work_folder = pathlib.Path('./output')
            os.makedirs(work_folder, exist_ok=True)
            if titrate:
                for f0_ratio in f0_ratios:
                    filename_prefix = f'{input_path.stem}_{f0_ratio}'
                    resamp_f0_mix = librosa.resample(f0_ratios[f0_ratio]['f0_mix'], orig_sr=sr, target_sr=orig_sr)
                    wavfile.write(work_folder.joinpath(filename_prefix + '_boosted.wav'), orig_sr, resamp_f0_mix)
            else:
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

        if eval_haaqi:
            if track_id not in tracks:
                tracks[track_id] = dict()
            scores = {'normal': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}
            for ag in test_ags:
                if titrate:
                    for f0_ratio in f0_ratios:
                        scores[ag][f'ref_v_{f0_ratio}_mix'] = {'score': run_haaqi(sarr, f0_ratios[f0_ratio]['f0_mix'], sr, sr, test_ags[ag]), **metadata}
                else:
                    scores[ag]['ref_v_mix'] = {'score': run_haaqi(sarr, f0_mix, sr, sr, test_ags[ag]), **metadata}
                scores[ag]['ref_v_f0'] = {'score': run_haaqi(sarr, f0_contour, sr, sr, test_ags[ag]), **metadata}
                # for score in scores[ag]:
                #     print(f'HAAQI evaluation score for {score} against audiogram_{ag}: {scores[ag][score]}')
            tracks[track_id].update({ 'haaqi': scores })

        if eval_spectral:
            if track_id not in tracks:
                tracks[track_id] = dict()
            features = {'normal': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}
            for ag in test_ags:
                features[ag]['ref'] = {**get_spectral_features(sarr, sr, test_ags[ag]), **metadata}
                if titrate:
                    for f0_ratio in f0_ratios:
                        features[ag][f'{f0_ratio}_mix'] = {**get_spectral_features(f0_ratios[f0_ratio]['f0_mix'], sr, test_ags[ag]), **metadata}
                else:
                    features[ag]['mix'] = {**get_spectral_features(f0_mix, sr, test_ags[ag]), **metadata}
                features[ag]['f0'] = {**get_spectral_features(f0_contour, sr, test_ags[ag]), **metadata}
                # for feature in features[ag]:
                #     print(f'Spectral features extracted for {feature} against audiogram_{ag}: {features[ag][feature]}')
            tracks[track_id].update({ 'spectral': features })
    
    if tracks:
        with open(f'evaluation_{dataset}.json', 'w') as json_file:
            json.dump(tracks, json_file, indent=4)

        