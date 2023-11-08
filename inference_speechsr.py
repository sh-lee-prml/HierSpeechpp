import os
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils

from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def SuperResoltuion(a, hierspeech):
    
    speechsr = hierspeech

    os.makedirs(a.output_dir, exist_ok=True)

    # Prompt load
    audio, sample_rate = torchaudio.load(a.input_speech)

    # support only single channel
    audio = audio[:1,:] 
    # Resampling
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window") 
    file_name = os.path.splitext(os.path.basename(a.input_speech))[0]
        ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
    with torch.no_grad():
        converted_audio = speechsr(audio.unsqueeze(1).cuda())
        converted_audio = converted_audio.squeeze()
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 0.999 * 32767.0
        converted_audio = converted_audio.cpu().numpy().astype('int16')

    file_name2 = "{}.wav".format(file_name)
    output_file = os.path.join(a.output_dir, file_name2)
    
    if a.output_sr == 48000:
        write(output_file, 48000, converted_audio)
    else:
        write(output_file, 24000, converted_audio)


def model_load(a):
    if a.output_sr == 48000:
        speechsr = SpeechSR48(h_sr48.data.n_mel_channels,
            h_sr48.train.segment_size // h_sr48.data.hop_length,
            **h_sr48.model).cuda()
        utils.load_checkpoint(a.ckpt_sr48, speechsr, None)
        speechsr.eval()
    else:
        # 24000 Hz
        speechsr = SpeechSR24(h_sr.data.n_mel_channels,
        h_sr.train.segment_size // h_sr.data.hop_length,
        **h_sr.model).cuda()
        utils.load_checkpoint(a.ckpt_sr, speechsr, None)
        speechsr.eval()
    return speechsr

def inference(a):
    
    speechsr = model_load(a) 
    SuperResoltuion(a, speechsr)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_speech', default='example/reference_4.wav')
    parser.add_argument('--output_dir', default='SR_results')
    parser.add_argument('--ckpt_sr', type=str, default='./speechsr24k/G_340000.pth')  
    parser.add_argument('--ckpt_sr48', type=str, default='./speechsr48k/G_100000.pth')  
    parser.add_argument('--output_sr', type=float, default=48000)
    a = parser.parse_args()

    global device, h_sr, h_sr48
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )


    inference(a)

if __name__ == '__main__':
    main()