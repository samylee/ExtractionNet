import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class ExtractionNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, init_weights=True, phase='train'):
        super(ExtractionNet, self).__init__()
        self._phase = phase
        self.features = self.make_layers(in_channels=in_channels, out_channels=num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)

        if self._phase == 'test':
            x = torch.flatten(x, 1)
            x = F.softmax(x, dim=-1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, in_channels=3, out_channels=1000):
        # conv: out_channels, kernel_size, stride, batchnorm, activate
        # maxpool: kernel_size stride
        params = [
            [64, 7, 2, True, 'leaky'],
            ['M', 2, 2],
            [192, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [128, 1, 1, True, 'leaky'],
            [256, 3, 1, True, 'leaky'],
            [256, 1, 1, True, 'leaky'],
            [512, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [256, 1, 1, True, 'leaky'],
            [512, 3, 1, True, 'leaky'],
            [256, 1, 1, True, 'leaky'],
            [512, 3, 1, True, 'leaky'],
            [256, 1, 1, True, 'leaky'],
            [512, 3, 1, True, 'leaky'],
            [256, 1, 1, True, 'leaky'],
            [512, 3, 1, True, 'leaky'],
            [512, 1, 1, True, 'leaky'],
            [1024, 3, 1, True, 'leaky'],
            ['M', 2, 2],
            [512, 1, 1, True, 'leaky'],
            [1024, 3, 1, True, 'leaky'],
            [512, 1, 1, True, 'leaky'],
            [1024, 3, 1, True, 'leaky'],
            [out_channels, 1, 1, False, 'leaky'], # classifier
            ['A']
        ]

        module_list = nn.ModuleList()
        for i, v in enumerate(params):
            modules = nn.Sequential()
            if v[0] == 'M':
                modules.add_module(f"maxpool_{i}", nn.MaxPool2d(kernel_size=v[1], stride=v[2], padding=int((v[1] - 1) // 2)))
            elif v[0] == 'A':
                modules.add_module(f"avgpool_{i}", nn.AdaptiveAvgPool2d((1, 1)))
            else:
                modules.add_module(
                    f"conv_{i}",
                    nn.Conv2d(
                        in_channels,
                        v[0],
                        kernel_size=v[1],
                        stride=v[2],
                        padding=(v[1] - 1) // 2,
                        bias=not v[3]
                    )
                )
                if v[3]:
                    modules.add_module(f"bn_{i}", nn.BatchNorm2d(v[0]))
                modules.add_module(f"act_{i}", nn.LeakyReLU(0.1) if v[4] == 'leaky' else nn.ReLU())
                in_channels = v[0]
            module_list.append(modules)
        return module_list


def load_darknet_weights(model, weights_path):
    # Open the weights file
    with open(weights_path, "rb") as f:
        # First five are header values
        header = np.fromfile(f, dtype=np.int32, count=4)
        header_info = header  # Needed to write header when saving weights
        seen = header[3]  # number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for module in model.features:
        if isinstance(module[0], nn.Conv2d):
            conv_layer = module[0]
            if isinstance(module[1], nn.BatchNorm2d):
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(
                    weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(
                    weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(
                    weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(
                    weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


if __name__ == '__main__':
    # load moel
    checkpoint_path = "extraction.weights"
    model = ExtractionNet(phase='test')
    load_darknet_weights(model, checkpoint_path)
    model.eval()

    # model input size
    net_size = 224

    # load img
    img = cv2.imread('dog.jpg')
    # img bgr2rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize img
    h, w, _ = img_rgb.shape
    new_h, new_w = h, w
    if w < h:
        new_h = (h * net_size) // w
        new_w = net_size
    else:
        new_w = (w * net_size) // h
        new_h = net_size
    img_resize = cv2.resize(img_rgb, (new_w, new_h))

    # crop img
    cut_w = (new_w - net_size) // 2
    cut_h = (new_h - net_size) // 2
    img_crop = img_resize[cut_h:cut_h + net_size, cut_w:cut_w + net_size, :]

    # float
    img_crop = torch.from_numpy(img_crop.transpose((2, 0, 1)))
    img_float = img_crop.float().div(255).unsqueeze(0)

    # forward
    result = model(img_float)

    print(result.topk(5))
