
import time
import math
import random

import threading
import hashlib
import pickle
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np



class cache:
    def __init__(self, function):
        self.function = function
        self.pickle_name = self.function.__name__

    def __call__(self, *args, **kwargs):
        m = hashlib.sha256()
        m.update(pickle.dumps((self.function.__name__, args, frozenset(kwargs.items()))))
        output_path = os.path.join('.cache', "%s_%s" % (m.hexdigest(), self.pickle_name))
        try:
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            data = self.function(*args, **kwargs)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        return data


def save_model(x, name):
    if isinstance(x, nn.DataParallel):
        torch.save(x.module.state_dict(), name)
    else:
        torch.save(x.state_dict(), name)

#
# class AsyncCall(object):
#     def __init__(self, fnc, callback=None):
#         self.Callable = fnc
#         self.Callback = callback
#         self.result = None
#
#     def __call__(self, *args, **kwargs):
#         self.Thread = threading.Thread(target=self.run, name=self.Callable.__name__, args=args, kwargs=kwargs)
#         self.Thread.start()
#         return self
#
#     def wait(self, timeout=None):
#         self.Thread.join(timeout)
#         if self.Thread.isAlive():
#             raise TimeoutError
#         else:
#             return self.result
#
#     def run(self, *args, **kwargs):
#         self.result = self.Callable(*args, **kwargs)
#         if self.Callback:
#             self.Callback(self.result)
#
#
# class AsyncMethod(object):
#     def __init__(self, fnc, callback=None):
#         self.Callable = fnc
#         self.Callback = callback
#
#     def __call__(self, *args, **kwargs):
#         return AsyncCall(self.Callable, self.Callback)(*args, **kwargs)


# def async_func(fnc=None, callback=None):
#     if fnc is None:
#         def add_async_callback(f):
#             return AsyncMethod(f, callback)
#         return add_async_callback
#     else:
#         return AsyncMethod(fnc, callback)


class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name):
        def register_fn(module):
            assert module_name not in self
            self[module_name] = module
            return module
        return register_fn


def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x)**2)


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()

MODELS = Registry()
ENCODERS = Registry()
GENERATORS = Registry()
MAPPINGS = Registry()
DISCRIMINATORS = Registry()



class Bool:
    def __init__(self):
        self.value = False

    def __bool__(self):
        return self.value
    __nonzero__ = __bool__

    def set(self, value):
        self.value = value


use_implicit_lreq = Bool()
use_implicit_lreq.set(True)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def make_tuple(x, n):
    if is_sequence(x):
        return x
    return tuple([x for _ in range(n)])


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gain=np.sqrt(2.0), lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.gain = gain
        self.lrmul = lrmul
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.in_features) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input):
        if not self.implicit_lreq:
            bias = self.bias
            if bias is not None:
                bias = bias * self.lrmul
            return F.linear(input, self.weight * self.std, bias)
        else:
            return F.linear(input, self.weight, self.bias)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transpose=False, transform_kernel=False, lrmul=1.0,
                 implicit_lreq=use_implicit_lreq):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_tuple(kernel_size, 2)
        self.stride = make_tuple(stride, 2)
        self.padding = make_tuple(padding, 2)
        self.output_padding = make_tuple(output_padding, 2)
        self.dilation = make_tuple(dilation, 2)
        self.groups = groups
        self.gain = gain
        self.lrmul = lrmul
        self.transpose = transpose
        self.fan_in = np.prod(self.kernel_size) * in_channels // groups
        self.transform_kernel = transform_kernel
        if transpose:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *self.kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.std = 0
        self.implicit_lreq = implicit_lreq
        self.reset_parameters()

    def reset_parameters(self):
        self.std = self.gain / np.sqrt(self.fan_in) * self.lrmul
        if not self.implicit_lreq:
            init.normal_(self.weight, mean=0, std=1.0 / self.lrmul)
        else:
            init.normal_(self.weight, mean=0, std=self.std / self.lrmul)
            setattr(self.weight, 'lr_equalization_coef', self.std)
            if self.bias is not None:
                setattr(self.bias, 'lr_equalization_coef', self.lrmul)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x):
        if self.transpose:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv_transpose2d(x, w * self.std, bias, stride=self.stride,
                                          padding=self.padding, output_padding=self.output_padding,
                                          dilation=self.dilation, groups=self.groups)
            else:
                return F.conv_transpose2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                          output_padding=self.output_padding, dilation=self.dilation,
                                          groups=self.groups)
        else:
            w = self.weight
            if self.transform_kernel:
                w = F.pad(w, (1, 1, 1, 1), mode='constant')
                w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25
            if not self.implicit_lreq:
                bias = self.bias
                if bias is not None:
                    bias = bias * self.lrmul
                return F.conv2d(x, w * self.std, bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            else:
                return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)


class ConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, gain=np.sqrt(2.0), transform_kernel=False, lrmul=1.0, implicit_lreq=use_implicit_lreq):
        super(ConvTranspose2d, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              output_padding=output_padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              gain=gain,
                                              transpose=True,
                                              transform_kernel=transform_kernel,
                                              lrmul=lrmul,
                                              implicit_lreq=implicit_lreq)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0), transpose=False):
        super(SeparableConv2d, self).__init__()
        self.spatial_conv = Conv2d(in_channels, in_channels, kernel_size, stride, padding, output_padding, dilation,
                                   in_channels, False, 1, transpose)
        self.channel_conv = Conv2d(in_channels, out_channels, 1, bias, 1, gain=gain)

    def forward(self, x):
        return self.channel_conv(self.spatial_conv(x))


class SeparableConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 bias=True, gain=np.sqrt(2.0)):
        super(SeparableConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              output_padding, dilation, bias, gain, True)





def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, last=False, fused_scale=True):
        super(EncodeBlock, self).__init__()
        self.conv_1 = Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(inputs, affine=False)
        self.blur = Blur(inputs)
        self.last = last
        self.fused_scale = fused_scale
        if last:
            self.dense = Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False)
        self.style_1 = Linear(2 * inputs, latent_size)
        if last:
            self.style_2 = Linear(outputs, latent_size)
        else:
            self.style_2 = Linear(2 * outputs, latent_size)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        m = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
        style_1 = torch.cat((m, std), dim=1)

        x = self.instance_norm_1(x)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))

            x = F.leaky_relu(x, 0.2)
            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(x.view(x.shape[0], x.shape[1]))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2

            x = F.leaky_relu(x, 0.2)

            m = torch.mean(x, dim=[2, 3], keepdim=True)
            std = torch.sqrt(torch.mean((x - m) ** 2, dim=[2, 3], keepdim=True))
            style_2 = torch.cat((m, std), dim=1)

            x = self.instance_norm_2(x)

            w1 = self.style_1(style_1.view(style_1.shape[0], style_1.shape[1]))
            w2 = self.style_2(style_2.view(style_2.shape[0], style_2.shape[1]))

        return x, w1, w2


class DiscriminatorBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False, fused_scale=True, dense=False):
        super(DiscriminatorBlock, self).__init__()
        self.conv_1 = Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        self.dense_ = dense
        self.fused_scale = fused_scale
        if self.dense_:
            self.dense = Linear(inputs * 4 * 4, outputs)
        else:
            if fused_scale:
                self.conv_2 = Conv2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_2 = Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        if self.last:
            x = minibatch_stddev_layer(x)

        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.dense_:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            if not self.fused_scale:
                x = downscale2d(x)
            x = x + self.bias_2
        x = F.leaky_relu(x, 0.2)

        return x


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                self.conv_1 = ConvTranspose2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_1 = Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_1 = Linear(latent_size, 2 * outputs, gain=1)

        self.conv_2 = Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_2 = Linear(latent_size, 2 * outputs, gain=1)

        self.layer = layer

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, s1, s2, noise):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x


class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = Conv2d(inputs, channels, 1, 1, 0, gain=0.03)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


# Default Encoder. E network
@ENCODERS.register("EncoderDefault")
class EncoderDefault(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(EncoderDefault, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, False, fused_scale=fused_scale)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


# For ablation only. Not used in default configuration
@ENCODERS.register("EncoderWithFC")
class EncoderWithFC(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(EncoderWithFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles, self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles, self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


@ENCODERS.register("EncoderWithStatistics")
class Encoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb: nn.ModuleList[FromRGB] = nn.ModuleList()
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = EncodeBlock(inputs, outputs, latent_size, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            #print("encode_block%d %s styles out: %d" % ((i + 1), millify(count_parameters(block)), inputs))
            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x, lod):
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def encode2(self, x, lod, blend):
        x_orig = x
        styles = torch.zeros(x.shape[0], 1, self.latent_size)

        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        x, s1, s2 = self.encode_block[self.layer_count - lod - 1](x)
        styles[:, 0] += s1 * blend + s2 * blend

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x, s1, s2 = self.encode_block[i](x)
            styles[:, 0] += s1 + s2

        return styles

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)

    def get_statistics(self, lod):
        rgb_std = self.from_rgb[self.layer_count - lod - 1].from_rgb.weight.std().item()
        rgb_std_c = self.from_rgb[self.layer_count - lod - 1].from_rgb.std

        layers = []
        for i in range(self.layer_count - lod - 1, self.layer_count):
            conv_1 = self.encode_block[i].conv_1.weight.std().item()
            conv_1_c = self.encode_block[i].conv_1.std
            conv_2 = self.encode_block[i].conv_2.weight.std().item()
            conv_2_c = self.encode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


# For ablation only. Not used in default configuration
@ENCODERS.register("EncoderNoStyle")
class EncoderNoStyle(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=512, channels=3):
        super(EncoderNoStyle, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(inputs, outputs, last=False, fused_scale=fused_scale, dense=i == self.layer_count - 1)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = Linear(inputs, latent_size, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x).view(x.shape[0], 1, x.shape[1])

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


# For ablation only. Not used in default configuration
@DISCRIMINATORS.register("DiscriminatorDefault")
class Discriminator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, channels=3):
        super(Discriminator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.from_rgb = nn.ModuleList()
        self.channels = channels

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[DiscriminatorBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            self.from_rgb.append(FromRGB(channels, inputs))

            fused_scale = resolution >= 128

            block = DiscriminatorBlock(inputs, outputs, i == self.layer_count - 1, fused_scale=fused_scale)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

        self.fc2 = Linear(inputs, 1, gain=1)

    def encode(self, x, lod):
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count - lod - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def encode2(self, x, lod, blend):
        x_orig = x
        x = self.from_rgb[self.layer_count - lod - 1](x)
        x = F.leaky_relu(x, 0.2)
        x = self.encode_block[self.layer_count - lod - 1](x)

        x_prev = F.avg_pool2d(x_orig, 2, 2)

        x_prev = self.from_rgb[self.layer_count - (lod - 1) - 1](x_prev)
        x_prev = F.leaky_relu(x_prev, 0.2)

        x = torch.lerp(x_prev, x, blend)

        for i in range(self.layer_count - (lod - 1) - 1, self.layer_count):
            x = self.encode_block[i](x)

        return self.fc2(x)

    def forward(self, x, lod, blend):
        if blend == 1:
            return self.encode(x, lod)
        else:
            return self.encode2(x, lod, blend)


@GENERATORS.register("GeneratorDefault")
class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        mul = 2 ** (self.layer_count - 1)

        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4))
        init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale, layer=i)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            self.decode_block.append(block)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, noise):
        x = self.const

        for i in range(lod + 1):
            x = self.decode_block[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x = self.to_rgb[lod](x)
        return x

    def decode2(self, styles, lod, blend, noise):
        x = self.const

        for i in range(lod):
            x = self.decode_block[i](x, styles[:, 2 * i + 0], styles[:, 2 * i + 1], noise)

        x_prev = self.to_rgb[lod - 1](x)

        x = self.decode_block[lod](x, styles[:, 2 * lod + 0], styles[:, 2 * lod + 1], noise)
        x = self.to_rgb[lod](x)

        needed_resolution = self.layer_to_resolution[lod]

        x_prev = F.interpolate(x_prev, size=needed_resolution)
        x = torch.lerp(x_prev, x, blend)

        return x

    def forward(self, styles, lod, blend, noise):
        if blend == 1:
            return self.decode(styles, lod, noise)
        else:
            return self.decode2(styles, lod, blend, noise)

    def get_statistics(self, lod):
        rgb_std = self.to_rgb[lod].to_rgb.weight.std().item()
        rgb_std_c = self.to_rgb[lod].to_rgb.std

        layers = []
        for i in range(lod + 1):
            conv_1 = 1.0
            conv_1_c = 1.0
            if i != 0:
                conv_1 = self.decode_block[i].conv_1.weight.std().item()
                conv_1_c = self.decode_block[i].conv_1.std
            conv_2 = self.decode_block[i].conv_2.weight.std().item()
            conv_2_c = self.decode_block[i].conv_2.std
            layers.append(((conv_1 / conv_1_c), (conv_2 / conv_2_c)))
        return rgb_std / rgb_std_c, layers


def minibatch_stddev_layer(x, group_size=4):
    group_size = min(group_size, x.shape[0])
    size = x.shape[0]
    if x.shape[0] % group_size != 0:
        x = torch.cat([x, x[:(group_size - (x.shape[0] % group_size)) % group_size]])
    y = x.view(group_size, -1, x.shape[1], x.shape[2], x.shape[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt((y ** 2).mean(dim=0) + 1e-8).mean(dim=[1, 2, 3], keepdim=True)
    y = y.repeat(group_size, 1, x.shape[2], x.shape[3])
    return torch.cat([x, y], dim=1)[:size]


image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 24

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


@GENERATORS.register("DCGANGenerator")
class DCGANGenerator(nn.Module):
    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(128, nc, 4, 2, 1),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x.view(x.shape[0], nz, 1, 1))


@ENCODERS.register("DCGANEncoder")
class DCGANEncoder(nn.Module):
    def __init__(self):
        super(DCGANEncoder, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(256, 24, 4, 1, 0),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.main(x)
        return x.view(x.shape[0], x.shape[1])


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


# For ablation only. Not used in default configuration
@MAPPINGS.register("MappingDefault")
class Mapping(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(Mapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.01)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)

    def forward(self, z):
        x = pixel_norm(z)

        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


# Used in default configuration. The D network
@MAPPINGS.register("MappingD")
class MappingD(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(MappingD, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = 2 * dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        # We select just one output. For compatibility with older models.
        # All other outputs are ignored
        # It is the same as if the last layer had one output.
        return x[:, 0, x.shape[2] // 2]


@MAPPINGS.register("MappingDNoStyle")
class MappingDNoStyle(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(MappingDNoStyle, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        return x[:, 0]


@MAPPINGS.register("MappingF")
class MappingF(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(MappingF, self).__init__()
        inputs = dlatent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = latent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        x = pixel_norm(x)

        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)


@ENCODERS.register("EncoderFC")
class EncoderFC(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(EncoderFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = Linear(28 * 28, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, latent_size)

    def encode(self, x, lod):
        x = F.interpolate(x, 28)
        x = x.view(x.shape[0], 28 * 28)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)
        x = F.leaky_relu(x, 0.2)

        return x

    def forward(self, x, lod, blend):
        return self.encode(x, lod)


@GENERATORS.register("GeneratorFC")
class GeneratorFC(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(GeneratorFC, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        self.fc_1 = Linear(latent_size, 1024)
        self.fc_2 = Linear(1024, 1024)
        self.fc_3 = Linear(1024, 28 * 28)

        self.layer_to_resolution = [28] * 10

    def decode(self, x, lod, blend_factor, noise):
        if len(x.shape) == 3:
            x = x[:, 0]  # no styles
        x.view(x.shape[0], self.latent_size)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, 28, 28)
        x = F.interpolate(x, 2 ** (2 + lod))
        return x

    def forward(self, x, lod, blend_factor, noise):
        return self.decode(x, lod, blend_factor, noise)


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="",
                 encoder="", z_regression=False):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression

        self.mapping_d = MAPPINGS["MappingD"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_f = MAPPINGS["MappingF"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_f(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_f.num_layers, 1)

                layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        discriminator_prediction = self.mapping_d(Z)
        return Z[:, :1], discriminator_prediction

    def forward(self, x, lod, blend_factor, d_train, ae):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True)

            Z, d_result_real = self.encode(rec, lod, blend_factor)

            assert Z.shape == s.shape

            if self.z_regression:
                Lae = torch.mean(((Z[:, 0] - z)**2))
            else:
                Lae = torch.mean(((Z - s.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(x, lod, blend_factor)

            _, d_result_fake = self.encode(Xp, lod, blend_factor)

            loss_d = discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)

            rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True)

            _, d_result_fake = self.encode(rec, lod, blend_factor)

            loss_g = generator_logistic_non_saturating(d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_d.parameters()) + list(self.mapping_f.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_d.parameters()) + list(other.mapping_f.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


class GenModel(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="", encoder="", z_regression=False):
        super(GenModel, self).__init__()

        self.layer_count = layer_count

        self.mapping_f = MAPPINGS["MappingF"](
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, blend_factor, z=None):
        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
        styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, True)
        return rec

    def forward(self, x):
        return self.generate(self.layer_count-1, 1.0, z=x)


#====================================================================================
def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION
    )
    model.cuda(local_rank)
    model.train()

    if local_rank == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER,
            z_regression=cfg.MODEL.Z_REGRESSION)
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        decoder = model.module.decoder
        encoder = model.module.encoder
        mapping_d = model.module.mapping_d
        mapping_f = model.module.mapping_f

        dlatent_avg = model.module.dlatent_avg
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_d = model.mapping_d
        mapping_f = model.mapping_f
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_f.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_d.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_d,
        'mapping_fl': mapping_f,
        'dlatent_avg': dlatent_avg
    }
    if local_rank == 0:
        model_dict['discriminator_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        model_dict['mapping_tl_s'] = model_s.mapping_d
        model_dict['mapping_fl_s'] = model_s.mapping_f

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size)

    if cfg.DATASET.SAMPLES_PATH != 'no_path':
        path = cfg.DATASET.SAMPLES_PATH
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(os.path.join(path, filename)))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
            sample = torch.stack(src)
    else:
        dataset.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, 32)
        sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
        sample = (sample / 127.5 - 1.)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
                                                                lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                lod2batch.lod,
                                                                2 ** lod2batch.get_lod_power2(),
                                                                2 ** lod2batch.get_lod_power2(),
                                                                lod2batch.get_blend_factor(),
                                                                len(dataset) * world_size))

        dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_per_GPU_batch_size())
        batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        for x_orig in tqdm(batches):
            i += 1
            with torch.no_grad():
                if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x_orig = x_orig.permute(0, 3, 1, 2)
                x_orig = (x_orig / 127.5 - 1.)

                blend_factor = lod2batch.get_blend_factor()

                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig

                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, needed_resolution)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            x.requires_grad = True

            encoder_optimizer.zero_grad()
            loss_d = model(x, lod2batch.lod, blend_factor, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            loss_g = model(x, lod2batch.lod, blend_factor, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            lae.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if local_rank == 0:
                betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                model_s.lerp(model, betta)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod_for_saving_model = lod2batch.lod
            lod2batch.step()
            if local_rank == 0:
                if lod2batch.is_time_to_save():
                    checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                if lod2batch.is_time_to_report():
                    save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s,
                                model.module if hasattr(model, "module") else model, cfg, encoder_optimizer,
                                decoder_optimizer)

        scheduler.step()

        if local_rank == 0:
            checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)
            save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s,
                        model.module if hasattr(model, "module") else model, cfg, encoder_optimizer, decoder_optimizer)

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()
