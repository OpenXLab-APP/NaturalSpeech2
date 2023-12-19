import gradio as gr
import argparse
import os
import torch
import soundfile as sf
import numpy as np

from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from encodec import EncodecModel
from encodec.utils import convert_audio
from utils.util import load_config

from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon

import torchaudio

from openxlab.model import download
download(model_repo='Amphion/NaturalSpeech2', model_name='pytorch_model', output='ckpts/ns2')


def build_codec(device):
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model = encodec_model.to(device=device)
    encodec_model.set_target_bandwidth(12.0)
    return encodec_model

def build_model(cfg, device):

    model = NaturalSpeech2(cfg.model)
    model.load_state_dict(
        torch.load(
            "ckpts/ns2/pytorch_model.bin",
            map_location="cpu",
        )
    )
    model = model.to(device=device)
    return model


def ns2_inference(
    prmopt_audio_path,
    text,
    diffusion_steps=100,
):
    try:
        import nltk
        nltk.download('cmudict')
    except:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ["WORK_DIR"] = "./"
    cfg = load_config("egs/tts/NaturalSpeech2/exp_config.json")

    model = build_model(cfg, device)
    codec = build_codec(device)

    ref_wav_path = prmopt_audio_path
    ref_wav, sr = torchaudio.load(ref_wav_path)
    ref_wav = convert_audio(
        ref_wav, sr, codec.sample_rate, codec.channels
    )
    ref_wav = ref_wav.unsqueeze(0).to(device=device)

    with torch.no_grad():
        encoded_frames = codec.encode(ref_wav)
        ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

    ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)

    symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
    phone2id = {s: i for i, s in enumerate(symbols)}
    id2phone = {i: s for s, i in phone2id.items()}
    
    lexicon = read_lexicon(cfg.preprocess.lexicon_path)
    phone_seq = preprocess_english(text, lexicon)


    phone_id = np.array(
        [
            *map(
                phone2id.get,
                phone_seq.replace("{", "").replace("}", "").split(),
            )
        ]
    )
    phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=device)


    x0, prior_out = model.inference(
        ref_code, phone_id, ref_mask, diffusion_steps
    )

    latent_ref = codec.quantizer.vq.decode(ref_code.transpose(0, 1))
    rec_wav = codec.decoder(x0)

    os.makedirs("result", exist_ok=True)
    sf.write(
        "result/{}.wav".format(prmopt_audio_path.split("/")[-1][:-4] + "_zero_shot_result"),
        rec_wav[0, 0].detach().cpu().numpy(),
        samplerate=24000,
    )

    result_file = "result/{}.wav".format(prmopt_audio_path.split("/")[-1][:-4] + "_zero_shot_result")
    return result_file


demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload a reference speech you want to clone timbre",
        type="filepath",
    ),
    gr.Textbox(
        value="Amphion is a toolkit that can speak, make sounds, and sing.",
        label="Text you want to generate",
        type="text",
    ),
    gr.Slider(
        10,
        1000,
        value=200,
        step=1,
        label="Diffusion Inference Steps",
        info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
    ),
]
demo_outputs = gr.Audio(label="")

demo = gr.Interface(
    fn=ns2_inference,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Amphion Zero-Shot TTS NaturalSpeech2"
)

if __name__ == "__main__":
    demo.launch()
