import os

from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.block import HybridBlock
import numpy as np

class ReLU6(HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

class HardSigmoid(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return self.act(x + 3.) / 6.

class HardSwish(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * self.act(x)


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class Activation(HybridBlock):
    """Activation function used in MobileNetV3"""
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)


def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1, active=True):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(nn.Activation('relu'))

def _add_conv_dw(out, dw_channels, channels, stride):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels)
    _add_conv(out, channels=channels)
    out.add(_SE(num_out=channels))
    
class _DWConv(HybridBlock):
    def __init__(self, out, dw_channels, channels, stride, **kwargs):
        super(_DWConv, self).__init__(**kwargs)
        self.conv1 = _add_conv_dw(out, dw_channels, channels, stride)
        self.SE = _SE(channels)
    def hybrid_forward(self, F, x):
        out = self.conv1(x)
        out = self.SE(out)
        return out

class _SE(HybridBlock):
    def __init__(self, num_out, ratio=4, \
                 act_func=("relu", "hard_sigmoid"), use_bn=False, prefix='', **kwargs):
        super(_SE, self).__init__(**kwargs)
        self.use_bn = use_bn
        num_mid = make_divisible(num_out // ratio)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=num_mid, \
                               kernel_size=1, use_bias=True)
        self.act1 = Activation(act_func[0])
        self.conv2 = nn.Conv2D(channels=num_out, \
                               kernel_size=1, use_bias=True)
        self.act2 = Activation(act_func[1])

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return F.broadcast_mul(x, out)


class LinearBottleneck(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            _add_conv(self.out, in_channels * t)
            _add_conv(self.out, in_channels * t, kernel=3, stride=stride,
                      pad=1, num_group=in_channels * t)
            _add_conv(self.out, channels, active=False)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out

class MobileNet(HybridBlock):
    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                _add_conv(self.features, channels=int(32), kernel=3, pad=1, stride=2)
                _add_conv_dw(self.features, dw_channels=32, channels=64, stride=1)
                _add_conv_dw(self.features, dw_channels=64, channels=128, stride=2)
                _add_conv_dw(self.features, dw_channels=128, channels=128, stride=1)
                _add_conv_dw(self.features, dw_channels=128, channels=256, stride=2)
                _add_conv_dw(self.features, dw_channels=256, channels=256, stride=1)
                _add_conv_dw(self.features, dw_channels=256, channels=512, stride=2)
                _add_conv_dw(self.features, dw_channels=512, channels=512, stride=1)
                _add_conv_dw(self.features, dw_channels=512, channels=512, stride=1)
                _add_conv_dw(self.features, dw_channels=512, channels=512, stride=1)
                _add_conv_dw(self.features, dw_channels=512, channels=512, stride=1)
                _add_conv_dw(self.features, dw_channels=512, channels=512, stride=1)
                _add_conv_dw(self.features, dw_channels=512, channels=1024, stride=2)
                _add_conv_dw(self.features, dw_channels=1024, channels=1024, stride=1)
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class MobileNetV2(nn.HybridBlock):
    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(32 * multiplier), kernel=3,
                          stride=2, pad=1)

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1] * 6 + [2, 1, 1] * 2 + [1]

                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.features.add(LinearBottleneck(in_channels=in_c, channels=c,
                                                       t=t, stride=s))

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features, last_channels)

                self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenet(multiplier=1.0, pretrained=False, ctx=mx.cpu(), **kwargs):
    net = MobileNet(**kwargs)
    return net

def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)

