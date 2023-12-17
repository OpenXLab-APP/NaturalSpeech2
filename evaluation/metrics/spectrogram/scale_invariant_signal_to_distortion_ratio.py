# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import librosa

import numpy as np

from torchmetrics import ScaleInvariantSignalDistortionRatio


def extract_si_sdr(audio_ref, audio_deg, fs=None, method="cut"):
    si_sdr = ScaleInvariantSignalDistortionRatio()

    if fs != None:
        audio_ref, _ = librosa.load(audio_ref, sr=fs)
        audio_deg, _ = librosa.load(audio_deg, sr=fs)
    else:
        audio_ref, fs = librosa.load(audio_ref)
        audio_deg, fs = librosa.load(audio_deg)

    if len(audio_ref) != len(audio_deg):
        if method == "cut":
            length = min(len(audio_ref), len(audio_deg))
            audio_ref = audio_ref[:length]
            audio_deg = audio_deg[:length]
        elif method == "dtw":
            _, wp = librosa.sequence.dtw(audio_ref, audio_deg, backtrack=True)
            audio_ref_new = []
            audio_deg_new = []
            for i in range(wp.shape[0]):
                ref_index = wp[i][0]
                deg_index = wp[i][1]
                audio_ref_new.append(audio_ref[ref_index])
                audio_deg_new.append(audio_deg[deg_index])
            audio_ref = np.array(audio_ref_new)
            audio_deg = np.array(audio_deg_new)
            assert len(audio_ref) == len(audio_deg)

    audio_ref = torch.from_numpy(audio_ref)
    audio_deg = torch.from_numpy(audio_deg)

    return si_sdr(audio_deg, audio_ref)
