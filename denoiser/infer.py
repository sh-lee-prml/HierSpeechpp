
import torch

def denoise(noisy_wav, model, hps):
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(noisy_wav.device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, hps.n_fft, hps.hop_size, hps.win_size, hps.compress_factor)
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(amp_g, pha_g, hps.n_fft, hps.hop_size, hps.win_size, hps.compress_factor)
    audio_g = audio_g / norm_factor
    return audio_g

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav