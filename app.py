import gradio as gr
import os
import torch



def build_codec():
    ...

def build_model():
    ...

def ns2_inference(
        prmopt_audio_path,
        text,
        diffusion_steps=100,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

demo_inputs = ...
demo_outputs = ...

demo = gr.Interface(
    fn=ns2_inference,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Amphion Zero-Shot TTS NaturalSpeech2"
)

if __name__ == "__main__":
    demo.launch()