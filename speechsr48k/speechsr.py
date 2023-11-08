import torch
from torch import nn
from torch.nn import functional as F
import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from torch.cuda.amp import autocast
import torchaudio
from einops import rearrange

from alias_free_torch import *
import activations

class AMPBlock0(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock0, self).__init__()
      
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers


        self.activations = nn.ModuleList([
            Activation1d(
                activation=activations.SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
        ])
  
    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock0

        self.resblocks = nn.ModuleList()
        for i in range(1):
            ch = upsample_initial_channel//(2**(i))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = activations.SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
      
            x = F.interpolate(x, int(x.shape[-1] * 3), mode='linear')
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.resblocks:
            l.remove_weight_norm()

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        n_fft, hop_length, win_length = resolution
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=torch.hann_window,
            normalized=True, center=False, pad_mode=None, power=None)

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(2, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(2,1), padding=(2, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(4,1), padding=(4, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, y):
        fmap = []

        x = self.spec_transform(y)  # [B, 2, Freq, Frames, 2]
        x = torch.cat([x.real, x.imag], dim=1)
        x = rearrange(x, 'b c w t -> b c t w')

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]
        resolutions = [[4096, 1024, 4096], [2048, 512, 2048], [1024, 256, 1024], [512, 128, 512], [256, 64, 256], [128, 32, 128]]

        discs = [DiscriminatorR(resolutions[i], use_spectral_norm=use_spectral_norm) for i in range(len(resolutions))]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self,

    spec_channels,
    segment_size,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size

    self.dec = Generator(1, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes)

  def forward(self, x):

    y = self.dec(x)
    return y
  @torch.no_grad()
  def infer(self, x, max_len=None):

    o = self.dec(x[:,:,:max_len])
    return o

