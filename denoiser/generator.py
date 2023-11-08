import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from denoiser.conformer import ConformerBlock
from denoiser.utils import get_padding_2d, LearnableSigmoid_2d
from pesq import pesq
from joblib import Parallel, delayed

class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlock(h, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class TSConformerBlock(nn.Module):
    def __init__(self, h):
        super(TSConformerBlock, self).__init__()
        self.h = h
        self.time_conformer = ConformerBlock(dim=h.dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=h.dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, h, num_tscblocks=4):
        super(MPNet, self).__init__()
        self.h = h
        self.num_tscblocks = num_tscblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSConformer = nn.ModuleList([])
        for i in range(num_tscblocks):
            self.TSConformer.append(TSConformerBlock(h))
        
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_mag, noisy_pha): # [B, F, T]
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
        x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)
        
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag*torch.cos(denoised_pha),
                                    denoised_mag*torch.sin(denoised_pha)), dim=-1)

        return denoised_mag, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g, h):

    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):

    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(), 
                            h.sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score
