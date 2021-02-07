import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from norse.torch.module import SequentialState    # Stateful sequential layers

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from loguru import logger

from models.common import Conv_S, Focus_S, BottleneckCSP_S, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = True  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # self.model, self.save = make_model(ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            #tried over here
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x_t:torch.tensor, augment=False, profile=False, state=None):
        # x: shape => (t, batchsize, h, w, depth)
        # output = []
        x = "torch.tensor"
        state = [None] * len(self.model) if state is None else state
        for t in range(x_t.size(0)): # x_t.size(0)):
            x, state = self.forward_standard(x_t[t, :, :, :, :], augment=augment, profile=profile, state=state)
            # logger.debug(f'forward one batch{len(x)}: [{len(x[0])}, {len(x[0][0])}, {len(x[0][0][0])}]')
            # output.append(x)

        return x

    def forward_standard(self, x, augment=False, profile=False, state=None):
        # x: shape => (batch_size, h, w, 3)
        '''
        logger.debug(f'input:{type(x)}')
        logger.debug(f'input:  {x.size()}')
        logger.debug(f'{augment}, {profile}')
        '''
        state = [None] * len(self.model) if state is None else state
        if augment:
            logger.critical("Is not altered to SNN")
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile, state=state)  # single-scale inference, train

    def forward_once(self, x, profile=False, state=None):
        # x: shape => (batch_size, h, w, 3)
        y, dt = [], []  # outputs
        # logger.debug(f'input:{type(x)}')
        # logger.debug(f'input:  {x.size()}')
        state = [None] * len(self.model) if state is None else state
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # x = [x,x,x,x,x]
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            # logger.info(f'{i} : {m.type} : {self.model.stateful_layers[i]}')
            if self.model.stateful_layers[i]:
                x, s = m(x, state[i])
                state[i] = s
            else:
                x = m(x)

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))

        return x, state

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def make_model(ch=None):
    if ch is None:
        ch = [3]

    anchors = [[10, 13, 16, 30, 33, 23],  # P3/8
               [30, 61, 62, 45, 59, 119],  # P4/16
               [116, 90, 156, 198, 373, 326]]  # P5/32
    nc = 1
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # backbone:
    layers.append(module_extender(Focus_S, [3, 32, 3], 1, -1, 0))  # 0-P1/2
    layers.append(module_extender(Conv, [32, 64, 3, 2], 1, -1, 1))  # 1-P2/4
    layers.append(module_extender(BottleneckCSP, [64, 64, 1], 1, -1, 2))
    layers.append(module_extender(Conv, [64, 128, 3, 2], 1, -1, 3))  # 3-P3/8
    layers.append(module_extender(BottleneckCSP, [128, 128, 3], 1, -1, 4))
    layers.append(module_extender(Conv, [128, 256, 3, 2], 1, -1, 5))  # 5-P4/16
    layers.append(module_extender(BottleneckCSP, [256, 256, 3], 1, -1, 6))
    layers.append(module_extender(Conv, [256, 512, 3, 2], 1, -1, 7))  # 7-P5/32
    layers.append(module_extender(SPP, [512, 512, [5, 9, 13]], 1, -1, 8))
    layers.append(module_extender(BottleneckCSP, [512, 512, 1, False], 1, -1, 9))  # 9

    # head
    layers.append(module_extender(Conv, [512, 256, 1, 1], 1, -1, 10))
    layers.append(module_extender(nn.Upsample, [None, 2, 'nearest'], 1, -1, 11))
    layers.append(module_extender(Concat, [1], 1, [-1, 6], 12))  # cat backbone P4
    layers.append(module_extender(BottleneckCSP, [512, 256, 1, False], 1, -1, 13))  # 13

    layers.append(module_extender(Conv, [256, 128, 1, 1], 1, -1, 14))
    layers.append(module_extender(nn.Upsample, [None, 2, 'nearest'], 1, -1, 15))
    layers.append(module_extender(Concat, [1], 1, [-1, 4], 16))  # cat backbone P3
    layers.append(module_extender(BottleneckCSP, [256, 128, 1, False], 1, -1, 17))  # 17 (P3/8-small)

    layers.append(module_extender(Conv, [128, 128, 3, 2], 1, -1, 18))
    layers.append(module_extender(Concat, [1], 1, [-1, 14], 19))  # cat head P4
    layers.append(module_extender(BottleneckCSP, [256, 256, 1, False], 1, -1, 20))  # 20 (P4/16-medium)

    layers.append(module_extender(Conv, [256, 256, 3, 2], 1, -1, 21))
    layers.append(module_extender(Concat, [1], 1, [-1, 10], 22))  # cat head P5
    layers.append(module_extender(BottleneckCSP, [512, 512, 1, False], 1, -1, 23))  # 23 (P5/32-medium)

    layers.append(module_extender(Detect, [2,
                                           [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                            [116, 90, 156, 198, 373, 326]],
                                           [128, 256, 512]],
                                  1,
                                  [17, 20, 23],
                                  24))  # Detect(P3, P4, P5)

    return SequentialState(*layers), sorted([6, 4, 14, 10, 17, 20, 23])


def module_extender(module: callable, args: list, numberr: int, fromm: Union[int,list], i: int):
    module_ = SequentialState(*[module(*args) for _ in range(numberr)]) if numberr > 1 else module(*args)  # module

    # logger.debug(module_.stateful_layers)
    t = str(module)[8:-2].replace('__main__.', '')  # module type
    np = sum([x.numel() for x in module_.parameters()])  # number params
    module_.i, module_.f, module_.type, module_.np = i, fromm, t, np  # attach index, 'from' index, type, number params

    return module_


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # [from, number, module, args]
    for i, (fromm, numberr, module, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print(module)
        module = eval(module) if isinstance(module, str) else module  # eval strings
        for j, a in enumerate(args):
            try:
                # print(a)
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        numberr = max(round(numberr * gd), 1) if numberr > 1 else numberr  # depth gain
        if module in [Conv_S, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, Focus_S, CrossConv, BottleneckCSP_S, BottleneckCSP, C3]:
            c1, c2 = ch[fromm], args[0]

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if module in [BottleneckCSP, BottleneckCSP_S, C3]:
                args.insert(2, numberr)
                numberr = 1
        elif module is nn.BatchNorm2d:
            args = [ch[fromm]]
        elif module is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in fromm])
        elif module is Detect:
            logger.debug(f'{fromm}, {args}')
            args.append([ch[x + 1] for x in fromm])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(fromm)
            logger.debug(f'{fromm}, {args}')
        else:
            c2 = ch[fromm]

        module_ = nn.Sequential(*[module(*args) for _ in range(numberr)]) if numberr > 1 else module(*args)  # module
        t = str(module)[8:-2].replace('__main__.', '')  # module type
        # print(f'layers.append(module_extender({t[14:] + ",":14} {str(args) + ",":22} {str(numberr) + ",":3}  {str(fromm) + ",":8}  {str(i):2}))')
        np = sum([x.numel() for x in module_.parameters()])  # number params
        module_.i, module_.f, module_.type, module_.np = i, fromm, t, np  # attach index, 'from' index, type, number params
        # print(module, module_,module_.i, module_.f, module_.type, module_.np)
        # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, fromm, numberr, np, t, args))  # print
        save.extend(x % i for x in ([fromm] if isinstance(fromm, int) else fromm) if x != -1)  # append to savelist
        layers.append(module_)
        ch.append(c2)

    return SequentialState(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
