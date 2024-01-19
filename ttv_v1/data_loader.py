
import os
import numpy as np
import torch
import torchaudio
import commons
import random
from torchaudio.transforms import MelSpectrogram, Spectrogram, MelScale
from utils import parse_filelist
np.random.seed(1234) 

class AudioDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """ 
    def __init__(self, config, training=True):
        super(AudioDataset, self).__init__()
        self.config = config
        self.data_ratio = config.data.train_data_ratio
        self.hop_length = config.data.hop_length
        self.training = training
        self.mel_length = config.train.segment_size // config.data.hop_length
        if self.training:
            self.segment_length = config.train.segment_size
        self.sample_rate = config.data.sampling_rate

        self.filelist_path = config.data.train_filelist_path \
            if self.training else config.data.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)
        self.f0_paths = parse_filelist(self.filelist_path.replace('_wav', '_f0'))
        self.token_paths = parse_filelist(self.filelist_path.replace('_wav', '_token'))
        self.w2v_paths = parse_filelist(self.filelist_path.replace('_wav', '_w2v'))
 
    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data

        return audio.squeeze(), sample_rate 
     
    def __getitem__(self, index): 
        audio_path = self.audio_paths[index]
        f0_path = self.f0_paths[index]
        text_path = self.token_paths[index]
        w2v_path = self.w2v_paths[index]
         
        audio, sample_rate = self.load_audio_to_torch(audio_path)
        f0 = torch.load(f0_path)
        w2v = torch.load(w2v_path, map_location='cpu')
        text_for_ctc = torch.load(text_path)
        text = self.add_blank_token(text_for_ctc)   
      
        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according config."""

        if not self.training: 
            return audio, f0, text, w2v 
        
        segment = torch.nn.functional.pad(audio, (0, self.segment_length - audio.shape[-1]), 'constant')
        length = torch.LongTensor([audio.shape[-1] // self.hop_length])

        f0_segment = torch.nn.functional.pad(f0, (0, self.segment_length // 80 - f0.shape[-1]), 'constant')
        w2v = torch.nn.functional.pad(w2v.squeeze(0), (0, self.segment_length // 320 - w2v.shape[-1]), 'constant')
 
        text_length = torch.LongTensor([text.shape[-1]])
        text = torch.nn.functional.pad(text, (0, 403 - text.shape[-1]), 'constant')
 
        text_ctc_length = torch.LongTensor([text_for_ctc.shape[-1]])
        text_for_ctc = torch.nn.functional.pad(text_for_ctc, (0, 201 - text_for_ctc.shape[-1]), 'constant')    
        
        return segment, f0_segment, length, text, text_length, w2v, text_for_ctc, text_ctc_length 

    def __len__(self):  
        return len(self.audio_paths)
    
    def add_blank_token(self, text): 
        text_norm = commons.intersperse(text, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    
class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]
