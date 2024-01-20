import os
import glob
import random
import argparse 
import numpy as np
from tqdm import tqdm 

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist 

import torchaudio.functional as F   
from text import text_to_sequence 

def main(args):
    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args)) 

def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    dset = DLoader(args.input_dir)
    d_sampler = torch.utils.data.distributed.DistributedSampler(
        dset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)

    collate_fn = Collate()
    d_loader = DataLoader(dset, num_workers=8, shuffle=False,
                          batch_size=1, pin_memory=True,
                          drop_last=False, collate_fn=collate_fn, sampler=d_sampler)

    for epoch in range(1):
        extract(d_loader, rank)

class DLoader():
    def __init__(self, input_dir):
        self.txt = glob.glob(os.path.join(input_dir, '**/*.txt'), recursive=True)
        print('txt num: ', len(self.txt))

    def __getitem__(self, index):
        return self.txt[index]

    def __len__(self):
        return len(self.txt)


class Collate():
    def __init__(self):
        pass
    def __call__(self, batch):
        return batch[0]


def extract(d_loader, rank):
    with torch.no_grad():
        for _, txt_path in enumerate(tqdm(d_loader)):      
            txt_filename = txt_path.replace("LibriTTS_txt", "LibriTTS_token").replace('.txt', '.pt')
            
            if not os.path.isfile(txt_filename):
                os.makedirs(os.path.dirname(txt_filename), exist_ok=True)      
                    
                with open(txt_path) as f:
                    text = f.readline().rstrip()

                try:
                    token = text_to_sequence(text, ["english_cleaners2"])
                except Exception as e:
                    print(f"Error processing file: {txt_path}")
                    print(f"{e}")
                    return  
                 
                token = torch.LongTensor(token)
                torch.save(token, txt_filename)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/workspace/raid/dataset/LibriTTS_text/train-clean-100') 
    a = parser.parse_args() 
    main(a)

