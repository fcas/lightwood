import math
import torch
from functools import reduce
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.helpers.log import log


class DefaultNet(torch.nn.Module):
    def __init__(self,
                 input_size: int = None,
                 output_size: int = None,
                 shape: list = None,
                 max_params: int = int(3e5)) -> None:

        super(DefaultNet, self).__init__()
        if input_size is not None and output_size is not None:
            self.input_size = input_size
            self.output_size = output_size
            shape = [self.input_size, max([self.input_size * 2, self.output_size * 2, 400]), self.output_size]
            # If the network is too big, shrink it
            if reduce(lambda x, y: x * y, shape) > max_params:
                hidden_size = math.floor(max_params / (self.input_size * self.output_size))

                if hidden_size > self.output_size:
                    shape = [self.input_size, hidden_size, self.output_size]
                else:
                    shape = [self.input_size, self.output_size]
        if shape is not None:
            layers = []
            for ind in range(len(shape) - 1):
                layers.append(torch.nn.Linear(shape[ind], shape[ind + 1]))
                if ind < len(shape) - 2:
                    layers.append(torch.nn.SELU())
        else:
            raise Exception('You must specify other a shape or an input and output size when creating a DefaultNet!')

        self.net = torch.nn.Sequential(*layers)
        self.to(get_devices()[0])

    def to(self, device: torch.device) -> torch.nn.Module:
        if 'cuda' not in str(torch.device) == 0:
            log.warning('Creating neural network on CPU, it will be significantly slower than using a GPU, consider using a GPU instead')
        self.net = self.net.to(device)

        self.device = device
        return self

    def forward(self, input):
        with LightwoodAutocast():
            output = self.net(input)

        return output
