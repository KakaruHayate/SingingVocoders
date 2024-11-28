import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np
import time


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
        m.bias.data.normal_(mean, std)


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_padding(kernel_size, dilation=1):
	return int((kernel_size*dilation - dilation)/2)


class SineGen(torch.nn.Module):
	""" Definition of sine generator
	SineGen(samp_rate, harmonic_num = 0,
			sine_amp = 0.1, noise_std = 0.003,
			voiced_threshold = 0,
			flag_for_pulse=False)
	samp_rate: sampling rate in Hz
	harmonic_num: number of harmonic overtones (default 0)
	sine_amp: amplitude of sine-waveform (default 0.1)
	noise_std: std of Gaussian noise (default 0.003)
	voiced_threshold: F0 threshold for U/V classification (default 0)
	flag_for_pulse: this SinGen is used inside PulseGen (default False)
	Note: when flag_for_pulse is True, the first time step of a voiced
		segment is always sin(np.pi) or cos(0)
	"""

	def __init__(self, samp_rate, harmonic_num=0,
				 sine_amp=0.1, noise_std=0.003,
				 voiced_threshold=0):
		super(SineGen, self).__init__()
		self.sine_amp = sine_amp
		self.noise_std = noise_std
		self.harmonic_num = harmonic_num
		self.dim = self.harmonic_num + 1
		self.sampling_rate = samp_rate
		self.voiced_threshold = voiced_threshold

	def _f02uv(self, f0):
		# generate uv signal
		uv = torch.ones_like(f0)
		uv = uv * (f0 > self.voiced_threshold)
		return uv

	def _f02sine(self, f0, upp):
		""" f0: (batchsize, length, dim)
			where dim indicates fundamental tone and overtones
		"""
		rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
		rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
		rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
		rad += F.pad(rad_acc, (0, 0, 1, -1))
		rad = rad.reshape(f0.shape[0], -1, 1)
		rad = torch.multiply(rad, torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1))
		rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
		rand_ini[..., 0] = 0
		rad += rand_ini
		sines = torch.sin(2 * np.pi * rad)
		return sines

	@torch.no_grad()
	def forward(self, f0, upp):
		""" sine_tensor, uv = forward(f0)
		input F0: tensor(batchsize=1, length, dim=1)
				  f0 for unvoiced steps should be 0
		output sine_tensor: tensor(batchsize=1, length, dim)
		output uv: tensor(batchsize=1, length, 1)
		"""
		f0 = f0.unsqueeze(-1)
		sine_waves = self._f02sine(f0, upp) * self.sine_amp
		uv = (f0 > self.voiced_threshold).float()
		uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
		noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
		noise = noise_amp * torch.randn_like(sine_waves)
		sine_waves = sine_waves * uv + noise
		return sine_waves


class SourceModuleHnNSF(torch.nn.Module):
	""" SourceModule for hn-nsf
	SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
				 add_noise_std=0.003, voiced_threshod=0)
	sampling_rate: sampling_rate in Hz
	harmonic_num: number of harmonic above F0 (default: 0)
	sine_amp: amplitude of sine source signal (default: 0.1)
	add_noise_std: std of additive Gaussian noise (default: 0.003)
		note that amplitude of noise in unvoiced is decided
		by sine_amp
	voiced_threshold: threhold to set U/V given F0 (default: 0)
	Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
	F0_sampled (batchsize, length, 1)
	Sine_source (batchsize, length, 1)
	noise_source (batchsize, length 1)
	uv (batchsize, length, 1)
	"""

	def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
				 add_noise_std=0.003, voiced_threshold=0):
		super(SourceModuleHnNSF, self).__init__()

		self.sine_amp = sine_amp
		self.noise_std = add_noise_std

		# to produce sine waveforms
		self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
								 sine_amp, add_noise_std, voiced_threshold)

		# to merge source harmonics into a single excitation
		self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
		self.l_tanh = torch.nn.Tanh()

	def forward(self, x, upp):
		sine_wavs = self.l_sin_gen(x, upp)
		sine_merge = self.l_tanh(self.l_linear(sine_wavs))
		return sine_merge


class UpsampleLayer(nn.Module):
	def __init__(self, channels=64, mel_bin=128, scale_factor=[8, 8, 2, 2, 2], hop_size=512, kernel_size=[17, 17, 5, 5, 5]):
		super(UpsampleLayer, self).__init__()
		if int(np.prod(scale_factor)) != hop_size:
			raise RuntimeError(f"product of scale factor not equal hop_size!!")
		else:
			pass
		self.mel_bin = mel_bin
		self.scale_factor = scale_factor
		self.conv1 = weight_norm(nn.Conv1d(in_channels=mel_bin, out_channels=mel_bin, kernel_size=5, stride=1, padding=get_padding(5, 1), bias=False))
		# self.conv1.apply(init_weights)
		self.conv2 = weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size[0]), stride=(1, 1), padding=(0, get_padding(kernel_size[0], 1)), bias=False))
		# self.conv2.apply(init_weights)
		self.conv3 = weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size[1]), stride=(1, 1), padding=(0, get_padding(kernel_size[1], 1)), bias=False))
		# self.conv3.apply(init_weights)
		self.conv4 = weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size[2]), stride=(1, 1), padding=(0, get_padding(kernel_size[2], 1)), bias=False))
		# self.conv4.apply(init_weights)
		self.conv5 = weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size[3]), stride=(1, 1), padding=(0, get_padding(kernel_size[3], 1)), bias=False))
		# self.conv5.apply(init_weights)
		self.conv6 = weight_norm(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size[4]), stride=(1, 1), padding=(0, get_padding(kernel_size[4], 1)), bias=False))
		# self.conv6.apply(init_weights)
		self.conv7 = weight_norm(nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(mel_bin, 1), stride=(1, 1), padding=0, bias=True))
		# self.conv7.apply(init_weights)

	def forward(self, m):
		# m:[B, M, T]
		x = self.conv1(m)
		x = x.unsqueeze(1)
		x = F.interpolate(x, scale_factor=[1, self.scale_factor[0]], mode='nearest')
		x = self.conv2(x)
		x = F.interpolate(x, scale_factor=[1, self.scale_factor[1]], mode='nearest')
		x = self.conv3(x)
		x = F.interpolate(x, scale_factor=[1, self.scale_factor[2]], mode='nearest')
		x = self.conv4(x)
		x = F.interpolate(x, scale_factor=[1, self.scale_factor[3]], mode='nearest')
		x = self.conv5(x)
		x = F.interpolate(x, scale_factor=[1, self.scale_factor[4]], mode='nearest')
		x = self.conv6(x)
		x = x.squeeze(1)
		x = x.unsqueeze(1)
		x = self.conv7(x)
		x = x.squeeze(2)

		return x
	
	def remove_weight_norm(self):
		for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
			remove_weight_norm(layer)


class WaveNetLayer(torch.nn.Module):
	def __init__(self, channels=64, kernel_size=5, block_num=0, layer_num=0):
		super(WaveNetLayer, self).__init__()
		self.block_num = block_num
		self.layer_num = layer_num
		if self.block_num == 0 and self.layer_num == 0:
			channels1 = 1
		else:
			channels1 = channels
		self.dilation = 2 ** layer_num
		self.conv1 = weight_norm(nn.Conv1d(in_channels=channels1, out_channels=channels1, kernel_size=kernel_size, stride=1, dilation=self.dilation, padding=get_padding(kernel_size, self.dilation), bias=True))
		# self.conv1.apply(init_weights)
		self.conv2 = weight_norm(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=self.dilation, padding=get_padding(kernel_size, self.dilation), bias=True))
		# self.conv2.apply(init_weights)
		self.conv3 = weight_norm(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, dilation=self.dilation, padding=get_padding(kernel_size, self.dilation), bias=True))
		# self.conv3.apply(init_weights)

	def forward(self, x1, x2):
		if self.layer_num == 0:
			res = 0
		else:
			res = x1
		x1 = self.conv1(x1)
		x2a = self.conv2(x2) + x1
		x2b = self.conv3(x2) + x1
		x = F.tanh(x2a) * F.sigmoid(x2b)
		
		return x + res

	def remove_weight_norm(self):
		for layer in [self.conv1, self.conv2, self.conv3]:
			remove_weight_norm(layer)


class WNGANGenerator(torch.nn.Module):
	def __init__(self, h):
		super(WNGANGenerator, self).__init__()
		self.h = h
		self.upp = int(np.prod(h.upsample_rates))
		self.m_source = SourceModuleHnNSF(
			sampling_rate=h.sampling_rate,
			harmonic_num=8
		)
		self.ups = UpsampleLayer(channels=h.res_layers_channel, mel_bin=h.num_mels, scale_factor=h.upsample_rates, hop_size=h.model_hop_size, kernel_size=h.wngan_upsample_kernel_sizes)
		self.res_layers = torch.nn.ModuleList()
		for i in range(h.res_blocks_num):
			for j in range(h.res_layers_num):
				layer = WaveNetLayer(
					channels=h.res_layers_channel,
					kernel_size=h.res_layers_kernel_size,
					block_num=i,
					layer_num=j
				)
				self.res_layers.append(layer)
		self.post_conv1 = weight_norm(nn.Conv1d(h.res_layers_channel, h.res_layers_channel, 1))
		# self.post_conv1.apply(init_weights)
		self.post_conv2 = weight_norm(nn.Conv1d(h.res_layers_channel, 1, 1))
		# self.post_conv2.apply(init_weights)

	def forward(self, x, f0):
		# x [B, T, M]
		# f0 [B, T]
		har_source = self.m_source(f0, self.upp).transpose(1, 2)
		x = self.ups(x)
		x_res = har_source
		for n in self.res_layers:
			s = x_res
			x_res = n(s, x)
		x = x_res
		x = F.relu(x)
		x = self.post_conv1(x)
		x = F.relu(x)
		x = self.post_conv2(x)
		
		return x
		
	def remove_weight_norm(self):
		remove_weight_norm(self.ups)
		for l in self.res_layers:
			l.remove_weight_norm()
		for layer in [self.post_conv1, self.post_conv2]:
			remove_weight_norm(layer)


class GeneratorTEST(torch.nn.Module):
	def __init__(self):
		super(GeneratorTEST, self).__init__()
		self.upp = int(np.prod([8, 8, 2, 2, 2]))
		self.m_source = SourceModuleHnNSF(
			sampling_rate=44100,
			harmonic_num=8
		)
		self.ups = UpsampleLayer(channels=64, mel_bin=128, scale_factor=[8, 8, 2, 2, 2], hop_size=512, kernel_size=[17, 17, 5, 5, 5])
		self.res_layers = torch.nn.ModuleList()
		for i in range(2):
			for j in range(8):
				layer = WaveNetLayer(
					channels=64,
					kernel_size=5,
					block_num=i,
					layer_num=j
				)
				self.res_layers.append(layer)
		self.post_conv1 = weight_norm(nn.Conv1d(64, 64, 1))
		self.post_conv2 = weight_norm(nn.Conv1d(64, 1, 1))

	def forward(self, x, f0):
		# x [B, T, M]
		# f0 [B, T]
		har_source = self.m_source(f0, self.upp).transpose(1, 2)
		x = self.ups(x.transpose(1, 2))
		x_res = har_source
		for n in self.res_layers:
			s = x_res
			x_res = n(s, x)
		x = x_res
		x = F.relu(x)
		x = self.post_conv1(x)
		x = F.relu(x)
		x = self.post_conv2(x)
		x = x.squeeze(1)
		
		return x
		
	def remove_weight_norm(self):
		for layer in [self.post_conv1, self.post_conv2]:
			remove_weight_norm(layer)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            x = torch.nan_to_num(x)

            fmap.append(x)

        x = self.conv_post(x)
        x = torch.nan_to_num(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class WNMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y):
        y_d_rs = []

        fmap_rs = []


        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)


        return y_d_rs, fmap_rs,


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding="valid")),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding="valid")),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding="valid")),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding="valid")),
                norm_f(Conv1d(1024, 1024, 5, 1, groups=1, padding="valid")),
                norm_f(Conv1d(1024, 1024, 3, 1, padding=1)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, kernel_size=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            x = torch.nan_to_num(x)
            fmap.append(x)

        x = self.conv_post(x)
        x = torch.nan_to_num(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class WNMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                # DiscriminatorS(),
                # DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y):
        y_d_rs = []

        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)

            y_d_r, fmap_r = d(y)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)


        return y_d_rs, fmap_rs,


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


if __name__ == "__main__":
	batch_size = 1
	T_frame = 600
	mel_bin = 128
	model = GeneratorTEST()
	print(model)

	input_tensor = torch.randn(batch_size, T_frame, mel_bin)
	input_tensor2 = torch.randn(batch_size, T_frame)
	if torch.cuda.is_available():
		model = model.cuda()
		input_tensor = input_tensor.cuda()
		input_tensor2 = input_tensor2.cuda()
	total_params = count_parameters(model)
	print(f"Total number of parameters: {total_params}")
	print("Input shape:", input_tensor.shape)  # [B, M, T]
	print("Input shape:", input_tensor2.shape)  # [B, M, T]
	n_repeats = 10
	total_time = 0.0
	for _ in range(n_repeats):
		start_time = time.time()
		output = model(input_tensor, input_tensor2)
		torch.cuda.synchronize()
		end_time = time.time()
		total_time += end_time - start_time
	avg_time = total_time / n_repeats
	print("Output shape:", output.shape)  # [B, M, n_samples]
	print(f"Time taken for forward pass(AVG.): {avg_time:.4f} seconds")
