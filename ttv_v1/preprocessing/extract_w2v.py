import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import random 
import os
import argparse
import glob
from tqdm import tqdm 
import transformers
 

class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7): 
        """we use the intermediate features of MMS.
           More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()
 
        self.mms = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m") 

        for param in self.mms.parameters():
            param.requires_grad = False
            param.grad = None
        self.mms.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        outputs = self.mms(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]   
        y = y.permute((0, 2, 1))   
        return y

def main(args):
    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0,100)
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
    w2v = Wav2vec2().cuda(rank)

    collate_fn = Collate()
    d_loader = DataLoader(dset, num_workers=8, shuffle=False,
                          batch_size=1, pin_memory=True,
                          drop_last=False, collate_fn=collate_fn, sampler=d_sampler)

    for epoch in range(1):
        extract(d_loader, w2v, rank)

class DLoader():
    def __init__(self, input_dir):
        self.wavs = [] 
        self.wavs += sorted(glob.glob(os.path.join(input_dir, '**/*.wav'), recursive=True))  
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        return self.wavs[index]

    def __len__(self):
        return len(self.wavs) 

class Collate():
    def __init__(self):
        pass
    def __call__(self, batch):
        return batch[0] 

def extract(d_loader, w2v, rank):  
    with torch.no_grad():
        for _, audio_path in enumerate(tqdm(d_loader)):  
            w2v_filename = audio_path.replace("LibriTTS_16k", "LibriTTS_w2v").replace('.wav', '.pt')
      
            if not os.path.isfile(w2v_filename):
                os.makedirs(os.path.dirname(w2v_filename), exist_ok=True)
        
                audio, sample_rate = torchaudio.load(audio_path)
                p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
                audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
 
                y_pad = torch.nn.functional.pad(audio, (40, 40), "reflect")
                w2v_x = w2v(y_pad.cuda(rank))
        
                torch.save(w2v_x, w2v_filename)  
 
if __name__ == '__main__': 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/workspace/raid/dataset/LibriTTS_16k')  
    a = parser.parse_args() 
    main(a)
