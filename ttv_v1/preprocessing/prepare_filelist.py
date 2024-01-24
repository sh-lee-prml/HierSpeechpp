import os
import glob
import argparse
import torchaudio
import torch
from tqdm import tqdm

def filter_audio_len(data_len, wav_min, wav_max):
    return wav_min <= data_len <= wav_max

def replace_path(path, old, new):
    return path.replace(old, new)

def make_filelist(file_list, filename):
    with open(filename, 'w') as file:
        for item in file_list:
            file.write(item + '\n')

def main(a, wav_min=32, wav_max=600, text_min=1, text_max=200):
    os.makedirs(a.output_dir, exist_ok=True)
    
    wavs = sorted(glob.glob(a.input_dir + '/**/*.wav', recursive=True))
    print("Wav num: ", len(wavs))

    valid_wavs, short_audio, long_audio = [], 0, 0

    # valid F0
    for wav in tqdm(wavs):
        f0_path = replace_path(wav, '16k', 'f0').replace('.wav', '.pt')
        f0_value = torch.load(f0_path)
        if f0_value.sum() != 0:
            valid_wavs.append(wav) 
    
    # valid wav 
    filtered_wavs = []
    for wav in tqdm(valid_wavs):
        data, _ = torchaudio.load(wav)
        data_len = data.size(-1) // 320
 
        if not filter_audio_len(data_len, wav_min, wav_max):
            if data_len < wav_min:
                short_audio += 1 
            else:
                long_audio += 1 
            continue  

        try:
            txt_path = replace_path(wav, '16k', 'token').replace('.wav', '.pt')
            txt = torch.load(txt_path)
        except:
            continue

        len_txt = txt.size(-1)
        if len_txt * 2 + 1 > data_len or not filter_audio_len(len_txt, text_min, text_max):
            continue

        filtered_wavs.append(wav)

    print(f"wav num: {len(wavs)}")
    print(f"valid F0 num: {len(valid_wavs)}")
    print(f"short wav num: {short_audio}")
    print(f"long wav num: {long_audio}")
    print(f"filtered num: {len(filtered_wavs)}")
 
    out_dir = os.path.basename(a.output_dir)
    make_filelist(filtered_wavs, f'{out_dir}/train_wav.txt')
    
    for i in ['f0', 'token', 'w2v']:
        filtered = [replace_path(wav, '16k', i).replace('.wav', '.pt') for wav in filtered_wavs]
        make_filelist(filtered, f'{out_dir}/train_{i}.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/workspace/raid/dataset/LibriTTS_16k/train-clean-100')
    parser.add_argument('-o', '--output_dir', default='/workspace/ha0/data_preprocess/filelist') 
       
    a = parser.parse_args() 
    main(a)
 
