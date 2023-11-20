import os
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils
from Mels_preprocess import MelSpectrogramFixed
from torch.nn import functional as F
from hierspeechpp_speechsynthesizer import (
    SynthesizerTrn, Wav2vec2
)
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from denoiser.generator import MPNet
from denoiser.infer import denoise

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

seed = 1111
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max':1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]] 
    f0 = np.vstack(f0s)
    return f0

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

def VC(a, hierspeech):
    
    net_g, speechsr, denoiser, mel_fn, w2v = hierspeech

    os.makedirs(a.output_dir, exist_ok=True)
   
    source_audio, sample_rate = torchaudio.load(a.source_speech)
    if sample_rate != 16000:
        source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000, resampling_method="kaiser_window")
    p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
    source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data
    file_name_s = os.path.splitext(os.path.basename(a.source_speech))[0]

    try:
        f0 = get_yaapt_f0(source_audio.numpy())
    except:
        f0 = np.zeros((1, 1, source_audio.shape[-1] // 80))
        f0 = f0.astype(np.float32)
        f0 = f0.squeeze(0) 

    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()

    y_pad = F.pad(source_audio, (40, 40), "reflect")
    x_w2v = w2v(y_pad.cuda())
    x_length = torch.LongTensor([x_w2v.size(2)]).to(device)

    # Prompt load
    target_audio, sample_rate = torchaudio.load(a.target_speech)
    # support only single channel
    target_audio = target_audio[:1,:] 
    # Resampling
    if sample_rate != 16000:
        target_audio = torchaudio.functional.resample(target_audio, sample_rate, 16000, resampling_method="kaiser_window") 
    if a.scale_norm == 'prompt':
        prompt_audio_max = torch.max(target_audio.abs())
    try:
        t_f0 = get_yaapt_f0(target_audio.numpy())
    except:
        t_f0 = np.zeros((1, 1, target_audio.shape[-1] // 80))
        t_f0 = t_f0.astype(np.float32)
        t_f0 = t_f0.squeeze(0)
    j = t_f0 != 0

    f0[ii] = ((f0[ii] * t_f0[j].std()) + t_f0[j].mean()).clip(min=0)
    denorm_f0 = torch.log(torch.FloatTensor(f0+1).cuda())
    # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
    ori_prompt_len = target_audio.shape[-1]
    p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
    target_audio = torch.nn.functional.pad(target_audio, (0, p), mode='constant').data

    file_name_t = os.path.splitext(os.path.basename(a.target_speech))[0]

    # If you have a memory issue during denosing the prompt, try to denoise the prompt with cpu before TTS 
    # We will have a plan to replace a memory-efficient denoiser 
    if a.denoise_ratio == 0:
        target_audio = torch.cat([target_audio.cuda(), target_audio.cuda()], dim=0)
    else:
        with torch.no_grad():
            denoised_audio = denoise(target_audio.squeeze(0).cuda(), denoiser, hps_denoiser)
        target_audio = torch.cat([target_audio.cuda(), denoised_audio[:,:target_audio.shape[-1]]], dim=0)

    
    target_audio = target_audio[:,:ori_prompt_len]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

    trg_mel = mel_fn(target_audio.cuda())

    trg_length = torch.LongTensor([trg_mel.size(2)]).to(device)
    trg_length2 = torch.cat([trg_length,trg_length], dim=0)


    with torch.no_grad():
        
        ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
        converted_audio = \
            net_g.voice_conversion_noise_control(x_w2v, x_length, trg_mel, trg_length2, denorm_f0, noise_scale=a.noise_scale_vc, denoise_ratio=a.denoise_ratio)
                
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

    file_name2 = "{}.wav".format(file_name_s+"_to_"+file_name_t)
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
    w2v = Wav2vec2().cuda()

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    net_g.load_state_dict(torch.load(a.ckpt))
    _ = net_g.eval()

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
    return net_g, speechsr, denoiser, mel_fn, w2v

def inference(a):
    
    hierspeech = model_load(a) 

    VC(a, hierspeech)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_speech', default='example/reference_2.wav')
    parser.add_argument('--target_speech', default='example/reference_1.wav')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--ckpt', default='./logs/hierspeechpp_eng_kor/hierspeechpp_v2_ckpt.pth')
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

    global device, hps, h_sr,h_sr48, hps_denoiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hps = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt)[0], 'config.json'))
    h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr)[0], 'config.json') )
    h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(a.ckpt_sr48)[0], 'config.json') )
    hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(a.denoiser_ckpt)[0], 'config.json'))

    inference(a)

if __name__ == '__main__':
    main()
