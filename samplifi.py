# Note: The transcribe() fucntion is lifted directly from Spotify's basic-pitch/inference.py
# The reason for copy-pasting instead of including the module (brittle, I know)
# is because basic-pitch doesn't provide any libraries for operating directly on
# tensors/arrays, but operates on audio files. To avoid too much writing/reading
# to disk, I copied the logic but allowed passing ndarrays as well.
#
# Author: afb8252@nyu.edu
# Project: samplifi

import os
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import RegularGridInterpolator as RGI
import librosa
import matplotlib.pyplot as plt
import pathlib
import pretty_midi
import argparse
import gin
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
from torch import from_numpy
from tensorflow import Tensor, signal, keras, saved_model, expand_dims
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from mir_eval.util import intervals_to_samples
from mir_eval.sonify import pitch_contour

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
import basic_pitch.note_creation as infer

from clarity.utils.audiogram import (
    Audiogram, AUDIOGRAM_REF, AUDIOGRAM_MILD, AUDIOGRAM_MODERATE, AUDIOGRAM_MODERATE_SEVERE
)
from clarity.evaluator.haaqi import compute_haaqi

import ddsp
import ddsp
import ddsp.training
from ddsp.training.postprocessing import (
    detect_notes, fit_quantile_transform
)

overlap = 30
overlap_len = overlap * FFT_HOP
hop_size = AUDIO_N_SAMPLES - overlap_len
window_len = 4096
hop_len = window_len//2
wtype = 'hann'
hbound = 13
min_freq = None
max_freq = None

f0_weight = 0.5
original_weight = 0.5

test_ags = {'ref': AUDIOGRAM_REF, 'mild': AUDIOGRAM_MILD, 'moderate': AUDIOGRAM_MODERATE, 'severe': AUDIOGRAM_MODERATE_SEVERE}
model_dir = pathlib.Path('./ddsp-models/pretrained')

def transcribe(sarr: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
    """Uses spotify's basic_pitch to get midi data from an audio file.

    Args:
        sarr: input signal array
        sr: sample rate

    Returns:
        midi array of transcribed signal
    """
    # From basic_pitch/inference.py:predict()
    model_or_model_path: Union[keras.Model, pathlib.Path, str] = ICASSP_2022_MODEL_PATH
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length: float = 127.70
    minimum_frequency: Optional[float] = min_freq
    maximum_frequency: Optional[float] = max_freq
    multiple_pitch_bends: bool = False
    melodia_trick: bool = True

    # Custom; extract tempo
    onset_env = librosa.onset.onset_strength(y=sarr, sr=sr)
    midi_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)

    # From basic_pitch/inference.py:run_inference()
    if isinstance(model_or_model_path, (pathlib.Path, str)):
        model = saved_model.load(str(model_or_model_path))
    else:
        model = model_or_model_path

    output = {"note": [], "onset": [], "contour": []}
    # From basic_pitch/inference.py:get_audio_input()
    slen = sarr.shape[0]
    # pad signal
    sarr = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), sarr])
    for window, window_time in window_audio(sarr):
        sarr_windowed = expand_dims(window, axis=0)
        for k, v in model(sarr_windowed).items():
            output[k].append(v)

    # back to run_inference()
    model_output = {k: unwrap_output(np.concatenate(output[k]),slen, overlap) for k in output}

    # From basic_pitch/inference.py:predict()
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))

    # calculate midi/note data from model output
    midi_data, _ = infer.model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,  # convert to frames
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=int(midi_tempo),
    )

    return midi_data

# From basic_pitch/inference.py:unwrap_output()
def unwrap_output(output: Tensor, audio_original_length: int, n_overlapping_frames: int) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    if type(output) is Tensor:
        raw_output = output.numpy()
    else:
        raw_output = output
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length


# From basic_pitch/inference.py:window_audio_file()
def window_audio(audio_original: np.ndarray[np.float32]) -> Iterable[Tuple[np.ndarray[np.float32], Dict[str, float]]]:
    """
    Pad appropriately an audio array, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(
                window,
                pad_width=[[0, AUDIO_N_SAMPLES - len(window)]],
            )
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        yield np.expand_dims(window, axis=-1), window_time

def get_f0s(marr: pretty_midi.PrettyMIDI, sarr_mags: np.ndarray, sr: int) -> np.ndarray:
    """Get f0s from Midi based on an array of times.

    Args:
        marr: midi object
        s_size: sample size in seconds
        sr: sample rate

    Returns:
        array of dicts with times and f0s for each instrument
    """
    f0s = np.array([])
    for inst in marr.instruments:
        if inst.is_drum:
            continue
        times = np.stack(np.fromiter(map(lambda a: np.array([a.start, a.end]), inst.notes), dtype=np.ndarray))
        freqs = np.fromiter(map(lambda a: librosa.midi_to_hz(a.pitch), inst.notes), dtype=sarr_mags.dtype)
        # TO-DO: figure out pitch bends

        # Sample size is the length of the audio in seconds (last entry in times) divided by the number of frames in STFT
        s_size = times[-1][1] / sarr_mags.shape[-1]
        s_times, s_freqs = intervals_to_samples(times, freqs, offset=0, sample_size=s_size, fill_value=0)
        f0s = np.append(f0s, dict({ 'inst': inst.name, 'times': np.array(s_times), 'freqs': np.array(s_freqs)}))
    return f0s


def f0_contour(sarr: np.ndarray, sarr_mags: np.ndarray, f0s: np.ndarray, sr: int) -> np.ndarray:
    """Create an f0 pitch contour from f0 midi data.

    Args:
        sarr: input signal array
        f0s: list of f0s from MIDI array
        sr: sample rate

    Returns:
        time series array of f0 contour
    """

    # Get time and freq grid for audio
    time_grid = librosa.times_like(sarr_mags, sr=sr, n_fft=window_len, hop_length=hop_len)
    freq_grid = librosa.fft_frequencies(sr=sr, n_fft=window_len)
    f_interp = RGI((time_grid, freq_grid), sarr_mags.T, bounds_error=False, fill_value=None)

    # Harmonics: First hbound harmonics and energy
    harmonics = np.arange(1, hbound)
    harmonic_frequencies = librosa.fft_frequencies(sr=sr, n_fft=window_len)

    full_contour = np.zeros_like(sarr)
    for f0 in f0s:
        energy = np.fromiter(map(lambda a, b: f_interp([a, b]), f0['times'], f0['freqs']), dtype=sarr.dtype)
        f0_contour = pitch_contour(f0['times'], f0['freqs'], amplitudes=energy, fs=sr, length=len(sarr))
        if len(energy) == sarr_mags.shape[-1]:
            harmonic_energy = librosa.f0_harmonics(sarr_mags, f0=f0['freqs'], harmonics=harmonics, freqs=harmonic_frequencies)
            for i, (factor, h_energy) in enumerate(zip(harmonics, harmonic_energy)):
                # Mix in harmonics
                f0_contour = f0_contour + pitch_contour(f0['times'], f0['freqs'] * factor, amplitudes=h_energy, fs=sr, length=len(sarr))
        else:
            print(f'[WARN] STFT shape ({sarr_mags.shape[-1]}) does not match frequency length ({len(energy)}), interpolating...')
            energy_interpolated = RGI((np.linspace(0, len(energy), len(sarr_mags)),), energy, bounds_error=False, fill_value=None)
            energy = energy_interpolated(np.arange(len(sarr_mags)))
            harmonic_energy = librosa.f0_harmonics(sarr_mags, f0=f0['freqs'], harmonics=harmonics, freqs=harmonic_frequencies)
            for i, (factor, h_energy) in enumerate(zip(harmonics, harmonic_energy)):
                h_energy_interpolated = RGI((np.linspace(0, len(h_energy), len(sarr_mags)),), h_energy, bounds_error=False, fill_value=None)
                h_energy = h_energy_interpolated(np.arange(len(sarr_mags)))
                # Mix in harmonics
                f0_contour += pitch_contour(f0['times'], f0['freqs'] * factor, amplitudes=h_energy, fs=sr, length=len(sarr))
        full_contour = full_contour + f0_contour

    # Normalize output and ensure bit depth matches input audio
    full_contour = librosa.util.normalize(f0_contour.astype(sarr.dtype, casting='same_kind'))

    return full_contour

def eval_haaqi(rsig: np.ndarray, psig: np.ndarray, rsr: int, psr: int, audiogram: Audiogram) -> int:
    """Run haaqi on reference and modified signal.

    Args:
        rsig: original input signal array
        psig: processed signal array
        sr: sample rate

    Returns:
        HAAQI V1 score
    """
    # Recommended to resample to 24kHz for HAAQI

    # Note: this seems to no longer be needed
    #rsig = librosa.resample(rsig, orig_sr=sr, target_sr=haaqi_sr)
    #psig = librosa.resample(psig, orig_sr=sr, target_sr=haaqi_sr)

    score = compute_haaqi(
        processed_signal = psig,
        reference_signal = rsig,
        processed_sample_rate = psr,
        reference_sample_rate = rsr,
        audiogram = audiogram,
        equalisation = 1,
        level1 = 65.0,
    )

    return score

def compute_timbre_transfer(sarr: np.ndarray, target_timbre: str, model_dir: pathlib.Path, sr: int) -> np.ndarray:
    """Run timbre transfer on an input signal

    Args:
        sarr: original input signal array
        target_timbre: One of ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone']
        sr: sample rate

    Returns:
        Timbre transferred signal
    """
    sarr = sarr[np.newaxis, :]

    # Compute features.
    audio_features = ddsp.training.metrics.compute_audio_features(sarr)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)

    # Set gin file
    gin_file = pathlib.Path(os.path.join(model_dir, 'operative_config-0.gin'))

    # Load the dataset statistics.
    dataset_stats = None
    dataset_stats_file = pathlib.Path(os.path.join(model_dir, 'dataset_statistics.pkl'))
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        if dataset_stats_file.is_file():
            with open(dataset_stats_file, 'rb') as f:
                dataset_stats = pickle.load(f)
    except Exception as err:
        print(f'Loading dataset statistics from pickle failed: {err}.')


    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in os.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = pathlib.Path(os.path.join(model_dir, ckpt_name))

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size_train = int(n_samples_train / time_steps_train)
    time_steps = int(sarr.shape[1] / hop_size_train)
    n_samples = time_steps * hop_size_train

    gin_params = [
        f'Harmonic.n_samples = {n_samples}',
        f'FilteredNoise.n_samples = {n_samples}',
        f'F0LoudnessPreprocessor.time_steps = {time_steps}',
        f'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)


    # Trim all input vectors to correct lengths
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        if not isinstance(audio_features[key], np.ndarray):
            audio_features[key] = audio_features[key].numpy()
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]


    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)
    print(f'Restoring model took {(time.time() - start_time)} seconds')

    threshold = 1
    adjust = True
    quiet = 30
    pitch_shift = -2
    loudness_shift = -4
    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
    mask_on = None

    if adjust and dataset_stats is not None:
        # Detect sections that are "on".
        mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                            audio_features['f0_confidence'],
                                            threshold)

        if np.any(mask_on):
            # Shift the pitch register.
            target_mean_pitch = dataset_stats['mean_pitch']
            pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
            mean_pitch = np.mean(pitch[mask_on])
            p_diff = target_mean_pitch - mean_pitch
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


            # Quantile shift the note_on parts.
            _, loudness_norm = fit_quantile_transform(
                audio_features['loudness_db'],
                mask_on,
                inv_quantile=dataset_stats['quantile_transform'])

            # Turn down the note_off parts.
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
            loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)

            audio_features_mod['loudness_db'] = loudness_norm

            # Manual Shifts.
            audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
            audio_features_mod = shift_f0(audio_features_mod, pitch_shift)

            af = audio_features if audio_features_mod is None else audio_features_mod

    # Run a batch of predictions.
    outputs = model(af, training=False)
    sarr_transferred = model.get_audio_from_outputs(outputs)

    return sarr_transferred



## Helper functions.
def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features['loudness_db'] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
    audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], .0, librosa.midi_to_hz(110.0))
    return audio_features



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, type=str, help='Input')
    parser.add_argument('--score', action='store_true', help='Compute HAAQI scores')
    parser.add_argument('--spec', action='store_true', help='Display spectrograms')
    parser.add_argument('--ddsp', type=str, help='What instrument to attempt timbre transfer')

    args = parser.parse_args()

    #print(ICASSP_2022_MODEL_PATH)

    input = pathlib.Path(args.input)
    score = args.score
    spec = args.spec
    target_inst = args.ddsp if args.ddsp else False
    work_folder = pathlib.Path('./output')
    os.makedirs(work_folder, exist_ok=True)

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
    f0_contour = f0_contour(sarr, sarr_mags, f0s, sr)

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
