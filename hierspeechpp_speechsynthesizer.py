import torch
from torch import nn
from torch.nn import functional as F
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

import torchaudio
from einops import rearrange
import transformers
import math
from styleencoder import StyleEncoder
import commons

from alias_free_torch import *
import activations

class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7, w2v='mms'):

        """we use the intermediate features of mms-300m.
           More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()

        if w2v == 'mms':
           self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")
        else:
           self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B x t)
        Returns:
            y: torch.Tensor of shape(B x C x t)
        """
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y

class ResidualCouplingBlock_Transformer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers=3,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.cond_block = torch.nn.Sequential(torch.nn.Linear(gin_channels, 4 * hidden_channels),
                                            nn.SiLU(), torch.nn.Linear(4 * hidden_channels, hidden_channels))

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer_Transformer_simple(channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):

    g = self.cond_block(g.squeeze(2))

    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

class PosteriorAudioEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.down_pre = nn.Conv1d(1, 16, 7, 1, padding=3)
    self.resblocks = nn.ModuleList()
    downsample_rates = [8,5,4,2]
    downsample_kernel_sizes = [17, 10, 8, 4]
    ch = [16, 32, 64, 128, 192]

    resblock = AMPBlock1
    resblock_kernel_sizes = [3,7,11]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
    self.num_kernels = 3
    self.downs = nn.ModuleList()
    for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
        self.downs.append(weight_norm(
            Conv1d(ch[i], ch[i+1], k, u, padding=(k-1)//2)))
    for i in range(4):
        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(resblock(ch[i+1], k, d, activation="snakebeta"))

    activation_post = activations.SnakeBeta(ch[i+1], alpha_logscale=True)
    self.activation_post = Activation1d(activation=activation_post)

    self.conv_post = Conv1d(ch[i+1], hidden_channels, 7, 1, padding=3)


    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels*2, out_channels * 2, 1)

  def forward(self, x, x_audio, x_mask, g=None):

    x_audio = self.down_pre(x_audio)

    for i in range(4):

        x_audio = self.downs[i](x_audio)

        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i*self.num_kernels+j](x_audio)
            else:
                xs += self.resblocks[i*self.num_kernels+j](x_audio)
        x_audio = xs / self.num_kernels

    x_audio = self.activation_post(x_audio)
    x_audio = self.conv_post(x_audio)

    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)

    x_audio = x_audio * x_mask

    x = torch.cat([x, x_audio], dim=1)

    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs

class PosteriorSFEncoder(nn.Module):
  def __init__(self,
      src_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()

    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre_source = nn.Conv1d(src_channels, hidden_channels, 1)
    self.pre_filter = nn.Conv1d(1, hidden_channels, kernel_size=9, stride=4, padding=4)
    self.source_enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers//2, gin_channels=gin_channels)
    self.filter_enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers//2, gin_channels=gin_channels)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers//2, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x_src, x_ftr, x_mask, g=None):

    x_src = self.pre_source(x_src) * x_mask
    x_ftr = self.pre_filter(x_ftr) * x_mask
    x_src = self.source_enc(x_src, x_mask, g=g)
    x_ftr = self.filter_enc(x_ftr, x_mask, g=g)
    x = self.enc(x_src+x_ftr, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs


class MelDecoder(nn.Module):
  def __init__(self,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      mel_size=20,
      gin_channels=0):
    super().__init__()

    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.conv_pre = Conv1d(hidden_channels, hidden_channels, 3, 1, padding=1)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)

    self.proj= nn.Conv1d(hidden_channels, mel_size, 1, bias=False)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

  def forward(self, x, x_mask, g=None):

    x = self.conv_pre(x*x_mask)
    if g is not None:
        x = x + self.cond(g)

    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    return x

class SourceNetwork(nn.Module):
  def __init__(self, upsample_initial_channel=256):
    super().__init__()

    resblock_kernel_sizes = [3,5,7]
    upsample_rates = [2,2]
    initial_channel = 192
    upsample_initial_channel = upsample_initial_channel
    upsample_kernel_sizes = [4,4]
    resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]

    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)

    self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
    resblock = AMPBlock1

    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        self.ups.append(weight_norm(
            ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                            k, u, padding=(k-u)//2)))

    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
        ch = upsample_initial_channel//(2**(i+1))
        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

    activation_post = activations.SnakeBeta(ch, alpha_logscale=True)
    self.activation_post = Activation1d(activation=activation_post)

    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    self.cond = Conv1d(256, upsample_initial_channel, 1)

    self.ups.apply(init_weights)


  def forward(self, x,  g):

    x = self.conv_pre(x) + self.cond(g)

    for i in range(self.num_upsamples):

      x = self.ups[i](x)
      xs = None
      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i*self.num_kernels+j](x)
        else:
          xs += self.resblocks[i*self.num_kernels+j](x)
      x = xs / self.num_kernels

    x = self.activation_post(x)
    ## Predictor
    x_ = self.conv_post(x)
    return x, x_
  
def remove_weight_norm(self):
    print('Removing weight norm...')
    for l in self.ups:
        remove_weight_norm(l)
    for l in self.resblocks:
        l.remove_weight_norm()

class DBlock(nn.Module):
  def __init__(self, input_size, hidden_size, factor):
    super().__init__()
    self.factor = factor
    self.residual_dense = weight_norm(Conv1d(input_size, hidden_size, 1))
    self.conv = nn.ModuleList([
        weight_norm(Conv1d(input_size, hidden_size, 3, dilation=1, padding=1)),
        weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2)),
        weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4)),
    ])
    self.conv.apply(init_weights)
  def forward(self, x):
    size = x.shape[-1] // self.factor

    residual = self.residual_dense(x)
    residual = F.interpolate(residual, size=size)

    x = F.interpolate(x, size=size)
    for layer in self.conv:
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      x = layer(x)

    return x + residual
  def remove_weight_norm(self):
    for l in self.conv:
        remove_weight_norm(l)

class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
      
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
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
    def __init__(self, initial_channel,  resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=256):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

          
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = activations.SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.downs = DBlock(upsample_initial_channel//8, upsample_initial_channel, 4)
        self.proj = Conv1d(upsample_initial_channel//8, upsample_initial_channel//2, 7, 1, padding=3)

    def forward(self, x, pitch, g=None):

        x = self.conv_pre(x) + self.downs(pitch) + self.cond(g)

        for i in range(self.num_upsamples):

            x = self.ups[i](x)

            if i == 0:
                pitch = self.proj(pitch)
                x = x + pitch

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
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.downs:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
  
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
        resolutions = [[2048, 512, 2048], [1024, 256, 1024], [512, 128, 512], [256, 64, 256], [128, 32, 128]]

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
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    gin_channels=256,
    prosody_size=20,
    uncond_ratio=0.,
    cfg=False,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.mel_size = prosody_size

    self.enc_p_l = PosteriorSFEncoder(1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow_l = ResidualCouplingBlock_Transformer(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)

    self.enc_p = PosteriorSFEncoder(1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.enc_q = PosteriorAudioEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock_Transformer(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)

    self.mel_decoder = MelDecoder(inter_channels,
                                     filter_channels,
                                     n_heads=2,
                                     n_layers=2,
                                     kernel_size=5,
                                     p_dropout=0.1,
                                     mel_size=self.mel_size,
                                     gin_channels=gin_channels)

    self.dec = Generator(inter_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.sn = SourceNetwork(upsample_initial_channel//2)
    self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=gin_channels)

    if cfg:

      self.emb = torch.nn.Embedding(1, 256)
      torch.nn.init.normal_(self.emb.weight, 0.0, 256 ** -0.5)
      self.null = torch.LongTensor([0]).cuda()
      self.uncond_ratio = uncond_ratio
    self.cfg = cfg
  @torch.no_grad()
  def infer(self, x_mel, w2v, length, f0):

    x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

    # Speaker embedding from mel (Style Encoder)
    g = self.emb_g(x_mel, x_mask).unsqueeze(-1)
    
    z, _, _ = self.enc_p_l(w2v, f0, x_mask, g=g)

    z = self.flow_l(z, x_mask, g=g, reverse=True)
    z = self.flow(z, x_mask, g=g, reverse=True)

    e, e_ = self.sn(z, g)
    o = self.dec(z, e, g=g)

    return o, e_
  @torch.no_grad()
  def voice_conversion(self, src, src_length, trg_mel, trg_length, f0, noise_scale = 0.333, uncond=False):

    trg_mask = torch.unsqueeze(commons.sequence_mask(trg_length, trg_mel.size(2)), 1).to(trg_mel.dtype)
    g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)

    y_mask = torch.unsqueeze(commons.sequence_mask(src_length, src.size(2)), 1).to(trg_mel.dtype)
    z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g)

    z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)*noise_scale) * y_mask

    z = self.flow_l(z, y_mask, g=g, reverse=True)
    z = self.flow(z, y_mask, g=g, reverse=True)

    if uncond:
        null_emb = self.emb(self.null) * math.sqrt(256)
        g = null_emb.unsqueeze(-1)

    e, _ = self.sn(z, g)
    o = self.dec(z, e, g=g)

    return o
  @torch.no_grad()
  def voice_conversion_noise_control(self, src, src_length, trg_mel, trg_length, f0, noise_scale = 0.333, uncond=False, denoise_ratio = 0):

    trg_mask = torch.unsqueeze(commons.sequence_mask(trg_length, trg_mel.size(2)), 1).to(trg_mel.dtype)
    g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)

    g_org, g_denoise = g[:1, :, :], g[1:, :, :]

    g_interpolation = (1-denoise_ratio)*g_org + denoise_ratio*g_denoise

    y_mask = torch.unsqueeze(commons.sequence_mask(src_length, src.size(2)), 1).to(trg_mel.dtype)
    z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g_interpolation)

    z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)*noise_scale) * y_mask

    z = self.flow_l(z, y_mask, g=g_interpolation, reverse=True)
    z = self.flow(z, y_mask, g=g_interpolation, reverse=True)

    if uncond:
        null_emb = self.emb(self.null) * math.sqrt(256)
        g = null_emb.unsqueeze(-1)

    e, _ = self.sn(z, g_interpolation)
    o = self.dec(z, e, g=g_interpolation)

    return o
  @torch.no_grad()  
  def f0_extraction(self, x_linear, x_mel, length, x_audio, noise_scale = 0.333):

    x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

    # Speaker embedding from mel (Style Encoder)
    g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

    # posterior encoder from linear spec. 
    _, m_q, logs_q= self.enc_q(x_linear, x_audio, x_mask, g=g)
    z = (m_q + torch.randn_like(m_q) * torch.exp(logs_q)*noise_scale)

    # Source Networks
    _, e_ = self.sn(z, g)

    return e_

