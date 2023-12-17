# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pymcd.mcd import Calculate_MCD


def extract_mcd(audio_ref, audio_deg, fs=None, mode="dtw_sl"):
    """Extract Mel-Cepstral Distance for a two given audio.
    Args:
        audio_ref: The given reference audio. It is an audio path.
        audio_deg: The given synthesized audio. It is an audio path.
        mode: "plain", "dtw" and "dtw_sl".
    """
    mcd_toolbox = Calculate_MCD(MCD_mode=mode)
    if fs != None:
        mcd_toolbox.SAMPLING_RATE = fs
    mcd_value = mcd_toolbox.calculate_mcd(audio_ref, audio_deg)

    return mcd_value
