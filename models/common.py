# This file contains modules common to various models

import math
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from typing import Union

# from norse.torch.module.lif import LIFFeedForwardLayer # Leaky integrate-and-fire
from norse.torch.module import LIFFeedForwardCell # Leaky integrate-and-fire
from norse.torch.module import SequentialState    # Stateful sequential layers
from norse.torch.functional import LIFParameters    # spiking lif parameters

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # logger.debug(type(self).__name__)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        # logger.debug(f'{type(self).__name__} fuse')
        return self.act(self.conv(x))

# default LIF Params
LIFPa = LIFParameters(
    tau_syn_inv=torch.as_tensor(1.0 / 5e-3),  # maybe disable it one run
    tau_mem_inv=torch.as_tensor(1.0 / 1e-2),
    v_leak=torch.as_tensor(0.0),
    v_th=torch.as_tensor(0.00001),
    v_reset=torch.as_tensor(0.0),  # changes at reset not used
    method="tent",
    alpha=torch.as_tensor(100.0),
    flags={"tau_syn_skip": False,
           "v_reset_skip": False},
)


class Conv_S(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, lifP=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_S, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if not lifP:
            lifP = LIFPa

        self.act = LIFFeedForwardCell(p=lifP, dt=0.001)

    def forward(self, input_tensor: torch.Tensor, state: Union[list, None] = None):
        # logger.debug(f'{type(self).__name__}, {self}')
        if isinstance(input_tensor, tuple):
            state = input_tensor[1]
            input_tensor = input_tensor[0]

        return self.act(self.bn(self.conv(input_tensor)), state)

    def fuseforward(self, x, state: Union[list, None] = None):
        # logger.debug(f'{type(self).__name__} fuse')
        return self.act(self.conv(x), state)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_S(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck_S, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_S(c1, c_, 1, 1)
        self.cv2 = Conv_S(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x, state=None):
        # logger.debug(type(self).__name__)
        if isinstance(x, tuple):
            state = x[1]
            x = x[0]
        if not isinstance(state, list):
            state = [None] * 2
        y1, state[0] = self.cv1(x, state[0])
        y2, state[1] = self.cv2(y1, state[1])
        return x + y2, state if self.add else y2, state


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        # logger.debug(type(self).__name__)
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP_S(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP_S, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_S(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act1 = LIFFeedForwardCell()
        self.m = SequentialState(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x, state=None):
        # logger.debug(type(self).__name__)
        if isinstance(x, tuple):
            state = x[1]
            x = x[0]
        if not isinstance(state, list):
            state = [None] * 4

        y1, state[0] = self.cv1(x,state[0])
        y1, state[1] = self.m(y1)
        y1 = self.cv3(y1)
        y2 = self.cv2(x)
        y3, state[2] = self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))), state[2])
        y3, state[3] = self.act1(y3, state[3])
        return  y3, state


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        # logger.debug("c3")
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        # logger.debug(type(self).__name__)
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPP_S(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP_S, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_S(c1, c_, 1, 1)
        self.cv2 = Conv_S(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x, state=None):
        # logger.debug(type(self).__name__)
        if isinstance(x, tuple):
            state = x[1]
            x = x[0]
        if not isinstance(state, list):
            state = [None] * 2
        x, state[0] = self.cv1(x, state[0])
        x, state[1] = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1), state[1])

        return x, state

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # logger.debug(type(self).__name__)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Focus_S(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, lifP=None):  # ch_in, ch_out, kernel, stride, padding, groups, activation, lifparameters
        super(Focus_S, self).__init__()
        self.conv = Conv_S(c1 * 4, c2, k, s, p, g, act, lifP)

    def forward(self, x, state: Union[list, None] = None):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # logger.debug("focus")
        if isinstance(x, tuple):
            state = x[1]
            x = x[0]

        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1), state)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # logger.debug(type(self).__name__)
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        # logger.debug("nms")
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1 = [], []  # image and inference shapes
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)  # open
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if show:
                img.show(f'Image {i}')  # show
            if pprint:
                print(str)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
