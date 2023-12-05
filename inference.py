import os
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils
from Mels_preprocess import MelSpectrogramFixed

from hierspeechpp_speechsynthesizer import (
    SynthesizerTrn
)
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from denoiser.generator import MPNet
from denoiser.infer import denoise

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def load_text(fp):
    with open(fp, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist
def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def add_blank_token(text):

    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def tts(text, a, hierspeech):
    
    net_g, text2w2v, speechsr, denoiser, mel_fn = hierspeech

    os.makedirs(a.output_dir, exist_ok=True)
    text = text_to_sequence(str(text), ["english_cleaners2"])
    token = add_blank_token(text).unsqueeze(0).cuda()
    token_length = torch.LongTensor([token.size(-1)]).cuda() 

    # Prompt load
    audio, sample_rate = torchaudio.load(a.input_prompt)

    # support only single channel
    audio = audio[:1,:] 
    # Resampling
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window") 
    if a.scale_norm == 'prompt':
        prompt_audio_max = torch.max(audio.abs())

    # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
    ori_prompt_len = audio.shape[-1]
    p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
    audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data

    file_name = os.path.splitext(os.path.basename(a.input_prompt))[0]

    # If you have a memory issue during denosing the prompt, try to denoise the prompt with cpu before TTS 
    # We will have a plan to replace a memory-efficient denoiser 
    if a.denoise_ratio == 0:
        audio = torch.cat([audio.cuda(), audio.cuda()], dim=0)
    else:
        with torch.no_grad():
            denoised_audio = denoise(audio.squeeze(0).cuda(), denoiser, hps_denoiser)
        audio = torch.cat([audio.cuda(), denoised_audio[:,:audio.shape[-1]]], dim=0)

    
    audio = audio[:,:ori_prompt_len]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

    src_mel = mel_fn(audio.cuda())

    src_length = torch.LongTensor([src_mel.size(2)]).to(device)
    src_length2 = torch.cat([src_length,src_length], dim=0)

    ## TTV (Text --> W2V, F0)
    with torch.no_grad():
        w2v_x, pitch = text2w2v.infer_noise_control(token, token_length, src_mel, src_length2, noise_scale=a.noise_scale_ttv, denoise_ratio=a.denoise_ratio)
   
        src_length = torch.LongTensor([w2v_x.size(2)]).cuda()  
        
        ## Pitch Clipping
        pitch[pitch<torch.log(torch.tensor([55]).cuda())]  = 0

        ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
        converted_audio = \
            net_g.voice_conversion_noise_control(w2v_x, src_length, src_mel, src_length2, pitch, noise_scale=a.noise_scale_vc, denoise_ratio=a.denoise_ratio)
                
        ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
        if a.output_sr == 48000: 
            converted_audio = speechsr(converted_audio)
        elif a.output_sr == 24000:
            converted_audio = speechsr(converted_audio)
        else:
            converted_audio = converted_audio

    converted_audio = converted_audio.squeeze()
    
    if a.scale_norm == 'prompt':
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * prompt_audio_max
    else:
        converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999 

    converted_audio = converted_audio.cpu().numpy().astype('int16')

    file_name2 = "{}.wav".format(file_name)
    output_file = os.path.join(a.output_dir, file_name2)
    
    if a.output_sr == 48000:
        write(output_file, 48000, converted_audio)
    elif a.output_sr == 24000:
        write(output_file, 24000, converted_audio)
    else:
        write(output_file, 16000, converted_audio)

def model_load(a):
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda()
    
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    net_g.load_state_dict(torch.load(a.ckpt))
    _ = net_g.eval()

    text2w2v = Text2W2V(hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps_t2w2v.model).cuda()
    text2w2v.load_state_dict(torch.load(a.ckpt_text2w2v))
    text2w2v.eval()

    if a.output_sr == 48000:
        speechsr = SpeechSR48(h_sr48.data.n_mel_channels,
            h_sr48.train.segment_size // h_sr48.data.hop_length,
            **h_sr48.model).cuda()
        utils.load_checkpoint(a.ckpt_sr48, speechsr, None)
        speechsr.eval()
       
    elif a.output_sr == 24000:
        speechsr = SpeechSR24(h_sr.data.n_mel_channels,
        h_sr.train.segment_size // h_sr.data.hop_length,
        **h_sr.model).cuda()
        utils.load_checkpoint(a.ckpt_sr, speechsr, None)
        speechsr.eval()
      
    else:
        speechsr = None
    
    denoiser = MPNet(hps_denoiser).cuda()
    state_dict = load_checkpoint(a.denoiser_ckpt, device)
    denoiser.load_state_dict(state_dict['generator'])
    denoiser.eval()
    return net_g, text2w2v, speechsr, denoiser, mel_fn

def inference(a):
    
    hierspeech = model_load(a) 
    # Input Text 
    text = load_text(a.input_txt)
    # text = "hello I'm hierspeech"
    
    tts(text, a, hierspeech)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_prompt', default='example/reference_4.wav')
    parser.add_argument('--input_txt', default='example/reference_4.txt')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--ckpt', default='./logs/hierspeechpp_eng_kor/hierspeechpp_v2_ckpt.pth')
    parser.add_argument('--ckpt_text2w2v', '-ct', help='text2w2v checkpoint path', default='./logs/ttv_libritts_v1/ttv_lt960_ckpt.pth')
    parser.add_argument('--ckpt_sr', type=str, default='./speechsr24k/G_340000.pth')  
    parser.add_argument('--ckpt_sr48', type=str, default='./speechsr48k/G_100000.pth')  
    parser.add_argument('--denoiser_ckpt', type=str, default='denoiser/g_best')
    parser.add_argument('--scale_norm', type=str, default='max')
    parser.add_argument('--output_sr', type=float, default=48000)
    parser.add_argument('--noise_scale_ttv', type=float,
                        default=0.333)
    parser.add_argument('--noise_scale_vc', type=float,
                        default=0.333)
    parser.add_argument('--denoise_ratio', type=float,
                        default=0.8)
    a = parser.parse_args()

    global device, hps, hps_t2w2v,h_sr,h_sr48, hps_denoiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt)[0], 'config.json'))
    hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_text2w2v)[0], 'config.json'))
    h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )
    hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(a.denoiser_ckpt)[0], 'config.json'))

    inference(a)

if __name__ == '__main__':
    main()
