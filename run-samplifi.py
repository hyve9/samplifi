import os
import sys
import numpy as np
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import pathlib
import argparse
import csv
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
)


from samplifi import (
    transcribe,
    get_f0s,
    get_f0_contour,
    eval_haaqi,
    compute_timbre_transfer,
    test_ags,
    window_len,
    hop_len,
    wtype,
    f0_weight,
    original_weight,
    model_dir,
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Input')
    parser.add_argument('--dataset', type=str, help='Run against a MIR dataset. (Run download-mir-dataset.py first to download the dataset.)')
    parser.add_argument('--score', action='store_true', help='Compute HAAQI scores')
    parser.add_argument('--spec', action='store_true', help='Display spectrograms')
    parser.add_argument('--ddsp', type=str, help='What instrument to attempt timbre transfer')

    args = parser.parse_args()

    if args.input and args.dataset:
        print('Cannot provide both --input and --dataset.')
        sys.exit(1)
    if args.input:
        input = pathlib.Path(args.input)
    elif args.dataset:
        dataset = args.dataset
    else:
        print('Must provide either --input or --dataset.')
        sys.exit(1)
    score = args.score
    spec = args.spec
    dataset = args.dataset
    target_inst = args.ddsp if args.ddsp else False
    work_folder = pathlib.Path('./output')
    os.makedirs(work_folder, exist_ok=True)

    if dataset:
        import mirdata
        #print(mirdata.list_datasets())
        if dataset not in mirdata.list_datasets():
            print('Dataset not found')
            sys.exit(1)
        data = mirdata.initialize(dataset, data_home=f'./mir_datasets/{dataset}')
        f = open('haaqi_scores.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['Audiogram', 'Track_ID', 'Comparison', 'Score', 'Instrument'])
        for key, track in data.load_tracks().items():
            print(key, track.audio_path)
            orig_sarr, orig_sr = librosa.load(track.audio_path, sr=None) # ndarray of amplitude values
            sarr = librosa.resample(orig_sarr, orig_sr=orig_sr, target_sr=AUDIO_SAMPLE_RATE)
            sr = AUDIO_SAMPLE_RATE

            # Get STFT for original audio # TO-DO: look at CWT or CQT
            sarr_stft = librosa.stft(sarr, n_fft=window_len, hop_length=hop_len, window=wtype)
            sarr_mags = np.abs(sarr_stft)

            # 1. Get midi array from input
            marr = transcribe(sarr, sr) # pretty midi object of instruments -> notes, bends, onsets and offsets

            # 2. Get sample times and f0s from midi array
            f0s = get_f0s(marr, sarr_mags, sr)

            # 3. Create f0 contour
            f0_contour = get_f0_contour(sarr, sarr_mags, f0s, sr)

            # 4. Mix into original signal
            f0_mix = f0_contour * f0_weight + sarr * original_weight
            rows = 3

            scores = {'ref': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}
            for ag in test_ags:
                # ref v ref isn't right... how do i use this thing?
                instrument = track.instrument if track.instrument else None
                scores[ag]['ref_v_ref'] = {'score': eval_haaqi(sarr, sarr, sr, sr, test_ags[ag]), 'instrument': instrument }
                scores[ag]['ref_v_f0'] = {'score': eval_haaqi(sarr, f0_contour, sr, sr, test_ags[ag]), 'instrument': instrument }
                scores[ag]['ref_v_mix'] = {'score': eval_haaqi(sarr, f0_mix, sr, sr, test_ags[ag]), 'instrument': instrument }
                for score in scores[ag]:
                    print(f'HAAQI evaluated score for {score} against audiogram_{ag}: {scores[ag][score]}')
                    writer.writerow([ag, track.track_id, score, scores[ag][score], instrument])
    elif input:
        orig_sarr, orig_sr = librosa.load(input, sr=None) # ndarray of amplitude values
        sarr = librosa.resample(orig_sarr, orig_sr=orig_sr, target_sr=AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE

        # Get STFT for original audio # TO-DO: look at CWT or CQT
        sarr_stft = librosa.stft(sarr, n_fft=window_len, hop_length=hop_len, window=wtype)
        sarr_mags = np.abs(sarr_stft)

        # 1. Get midi array from input
        marr = transcribe(sarr, sr) # pretty midi object of instruments -> notes, bends, onsets and offsets

        # 2. Get sample times and f0s from midi array
        f0s = get_f0s(marr, sarr_mags, sr)

        # 3. Create f0 contour
        f0_contour = get_f0_contour(sarr, sarr_mags, f0s, sr)

        # 4. Mix into original signal
        f0_mix = f0_contour * f0_weight + sarr * original_weight
        rows = 3

        # 4.1. Try timbre transfer
        if target_inst:
            timbre_transfer = compute_timbre_transfer(f0_contour, target_inst, model_dir, sr)
            #Fix shape
            timbre_transfer = timbre_transfer.numpy()[0]
            rows += 1

        if score:
            scores = {'ref': dict(), 'mild': dict(), 'moderate': dict(), 'severe': dict()}
            for ag in test_ags:
                # ref v ref isn't right... how do i use this thing?
                scores[ag]['ref_v_ref'] = eval_haaqi(sarr, sarr, sr, sr, test_ags[ag])
                scores[ag]['ref_v_f0'] = eval_haaqi(sarr, f0_contour, sr, sr, test_ags[ag])
                scores[ag]['ref_v_mix'] = eval_haaqi(sarr, f0_mix, sr, sr, test_ags[ag])
                if target_inst:
                    scores[ag]['ref_v_ddsp'] = eval_haaqi(sarr, timbre_transfer, sr, sr, test_ags[ag])
                for score in scores[ag]:
                    print(f'HAAQI evaluated score for {score} against audiogram_{ag}: {scores[ag][score]}')


        if spec:
            fig, ax = plt.subplots(nrows=rows, sharex=True)

            img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(sarr_stft), ref=np.max), y_axis='log', x_axis='time', ax=ax[0], n_fft=window_len, hop_length=hop_len)
            f0_contour_stft = librosa.stft(f0_contour, n_fft=window_len, hop_length=hop_len, window=wtype)
            img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(f0_contour_stft), ref=np.max), y_axis='log', x_axis='time', ax=ax[1], n_fft=window_len, hop_length=hop_len)
            f0_mix_stft = librosa.stft(f0_mix, n_fft=window_len, hop_length=hop_len, window=wtype)
            img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(f0_mix_stft), ref=np.max), y_axis='log', x_axis='time', ax=ax[2], n_fft=window_len, hop_length=hop_len)
            ax[0].set(title='Original')
            ax[0].label_outer()
            ax[1].set(title='f0 Only')
            ax[1].label_outer()
            ax[2].set(title='Boosted')
            ax[2].label_outer()
            if target_inst:
                timbre_transfer_stft = librosa.stft(timbre_transfer, n_fft=window_len, hop_length=hop_len, window=wtype)
                img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(timbre_transfer_stft), ref=np.max), y_axis='log', x_axis='time', ax=ax[3], n_fft=window_len, hop_length=hop_len)
                ax[3].set(title='Tone Transfer')
                ax[3].label_outer()

            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.show()


        # 5. Resample to original rate
        resamp_f0_mix = librosa.resample(f0_mix, orig_sr=sr, target_sr=orig_sr)
        resamp_f0_contour = librosa.resample(f0_contour, orig_sr=sr, target_sr=orig_sr)
        if target_inst:
            resamp_timbre_transfer = librosa.resample(timbre_transfer, orig_sr=sr, target_sr=orig_sr)

        marr.write(str(work_folder.joinpath(input.stem + '.mid')))
        wavfile.write(work_folder.joinpath(input.stem + '_f0.wav'), orig_sr, resamp_f0_contour)
        wavfile.write(work_folder.joinpath(input.stem + '_boosted.wav'), orig_sr, resamp_f0_mix)
        if target_inst:
            wavfile.write(work_folder.joinpath(input.stem + '_timbre_transfer.wav'), orig_sr, resamp_timbre_transfer)