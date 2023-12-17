# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import librosa

import numpy as np

from torchmetrics import PearsonCorrCoef

from utils.util import JsonHParams
from utils.f0 import get_f0_features_using_parselmouth, get_pitch_sub_median


def extract_fpc(
    audio_ref,
    audio_deg,
    fs=None,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
    pitch_bin=256,
    pitch_min=50,
    pitch_max=1100,
    need_mean=True,
    method="dtw",
):
    """Compute F0 Pearson Distance (FPC) between the predicted and the ground truth audio.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    fs: sampling rate.
    hop_length: hop length.
    f0_min: lower limit for f0.
    f0_max: upper limit for f0.
    pitch_bin: number of bins for f0 quantization.
    pitch_max: upper limit for f0 quantization.
    pitch_min: lower limit for f0 quantization.
    need_mean: subtract the mean value from f0 if "True".
    method: "dtw" will use dtw algorithm to align the length of the ground truth and predicted audio.
            "cut" will cut both audios into a same length according to the one with the shorter length.
    """
    # Initialize method
    pearson = PearsonCorrCoef()

    # Load audio
    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    # Initialize config
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max
    cfg.pitch_bin = pitch_bin
    cfg.pitch_max = pitch_max
    cfg.pitch_min = pitch_min

    # Compute f0
    f0_ref = get_f0_features_using_parselmouth(
        audio_ref,
        cfg,
    )[0]

    f0_deg = get_f0_features_using_parselmouth(
        audio_deg,
        cfg,
    )[0]

    # Subtract mean value from f0
    if need_mean:
        f0_ref = torch.from_numpy(f0_ref)
        f0_deg = torch.from_numpy(f0_deg)

        f0_ref = get_pitch_sub_median(f0_ref).numpy()
        f0_deg = get_pitch_sub_median(f0_deg).numpy()

    # Avoid silence
    min_length = min(len(f0_ref), len(f0_deg))
    if min_length <= 1:
        return 1

    # F0 length alignment
    if method == "cut":
        length = min(len(f0_ref), len(f0_deg))
        f0_ref = f0_ref[:length]
        f0_deg = f0_deg[:length]
    elif method == "dtw":
        _, wp = librosa.sequence.dtw(f0_ref, f0_deg, backtrack=True)
        f0_gt_new = []
        f0_pred_new = []
        for i in range(wp.shape[0]):
            gt_index = wp[i][0]
            pred_index = wp[i][1]
            f0_gt_new.append(f0_ref[gt_index])
            f0_pred_new.append(f0_deg[pred_index])
        f0_ref = np.array(f0_gt_new)
        f0_deg = np.array(f0_pred_new)
        assert len(f0_ref) == len(f0_deg)

    # Convert to tensor
    f0_ref = torch.from_numpy(f0_ref)
    f0_deg = torch.from_numpy(f0_deg)

    return pearson(f0_ref, f0_deg).numpy().tolist()
