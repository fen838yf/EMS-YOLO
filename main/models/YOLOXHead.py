import torch
import torch.nn as nn
from models.common import Conv,GhostConv,DWConv,RepConv,ConvL


import pkg_resources as pkg
def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, s
    return result


class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80, width=1.0, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.merge = Conv(ch, 256 * width, 1, 1)

        # 修改
        self.cls_convs1 = DWConv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs1 = GhostConv(256 * width, 256 * width, 3, 1, 1)

        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        # 分类=3x3conv + 3x3conv + 1x1convpred
        x1 = self.cls_convs1(x)
        # x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        # 回归=3x3conv（共享） + 3x3conv（共享） + 1x1pred
        x2 = self.reg_convs1(x)
        # x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        # 置信度=3x3conv（共享）+ 3x3conv（共享） + 1x1pred
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out


class YOLOXHead(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), Decoupled=False, ch=()):  # detection layer
        super(YOLOXHead, self).__init__()
        self.decoupled = Decoupled
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(DecoupledHead(x, nc, 1, anchors) for x in ch)

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
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=torch.float32, device=z.device)
        box @= convert_matrix
        return (box, score)



