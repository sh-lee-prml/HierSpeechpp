import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import random 
import commons
import utils

from ttv_v1.data_loader import AudioDataset, MelSpectrogramFixed
from ttv_v1.t2w2v_transformer import SynthesizerTrn 
from losses import kl_loss 

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0,100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_windows
    ).cuda(rank)

    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size, num_workers=32,
        sampler=train_sampler, drop_last=True
    )

    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)

    if rank == 0:
        num_param = get_param_num(net_g)
        print('Number of Total Parameters:', num_param)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps) 

    net_g = DDP(net_g, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g) 
        global_step = (epoch_str - 1) * len(train_loader)
    except: 
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2) 
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, mel_fn], optim_g, scaler, scheduler_g,
                               [train_loader, eval_loader], logger, [writer, writer_eval], n_gpus)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, mel_fn], optim_g, scaler,  scheduler_g,
                               [train_loader, None], None, None, n_gpus)
        scheduler_g.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, schedulers, loaders, logger, writers, n_gpus):
    net_g,  mel_fn = nets
    optim_g = optims
    scheduler_g = schedulers
    train_loader, eval_loader = loaders

    if writers is not None:
        writer, writer_eval = writers 
     
    global global_step 
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)
    net_g.train()     

    for batch_idx, (y, f0, length, text, text_length, w2v, text_for_ctc, text_ctc_length) in enumerate(train_loader):
        length = length.cuda(rank, non_blocking=True).squeeze()
        max_len = length.max()
        y = y[:, :max_len*320]
        w2v = w2v[:, :, :max_len]
        f0 = f0[:, :max_len*4]

        y = y.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        w2v = w2v.cuda(rank, non_blocking=True) 
        text_length = text_length.cuda(rank, non_blocking=True).squeeze()

        max_text_length = text_length.max()
        text = text[:,:max_text_length]
        text = text.cuda(rank, non_blocking=True) 
        text_ctc_length = text_ctc_length.cuda(rank, non_blocking=True).squeeze()

        max_text_ctc_length = text_ctc_length.max()
        text_for_ctc = text_for_ctc[:,:max_text_ctc_length]
        text_for_ctc = text_for_ctc.cuda(rank, non_blocking=True)

        mel = mel_fn(y) 
        f0 = torch.log(f0+1)

        with autocast(enabled=hps.train.fp16_run):
            w2v_x = w2v 
            w2v_predicted, (z, z_p, m_q, logs_q), (m_p, logs_p), mask, pitch_predicted, ids_slice, l_length, phoneme_predicted = \
                            net_g(w2v_x, length, text, text_length, mel)
            
            f0 = commons.slice_segments_audio(f0.squeeze(1), ids_slice * 4, 240)  
            loss_dur = torch.sum(l_length)
            loss_w2v = F.l1_loss(w2v_x, w2v_predicted) * hps.train.c_mel 
            loss_f0 = F.l1_loss(f0, pitch_predicted.squeeze(1))
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, mask) * hps.train.c_kl

            phoneme_predicted = phoneme_predicted.permute(2, 0, 1).log_softmax(2)
            loss_phoneme_prediction = F.ctc_loss(phoneme_predicted, text_for_ctc, length, text_ctc_length) 
            loss_gen_all = loss_w2v + loss_kl + loss_f0 * hps.train.c_f0 + loss_dur + loss_phoneme_prediction

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_w2v]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all,  "learning_rate": lr,
                                "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    {"loss/g/w2v": loss_w2v, "loss/g/kl": loss_kl, "loss/g/f0": loss_f0,"loss/g/dur": loss_dur,"loss/g/ctc": loss_phoneme_prediction})

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache() 

                if global_step % hps.train.save_interval == 0:
                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))  
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))
 
if __name__ == "__main__":
    main()
