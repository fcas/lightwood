import torch
from torch import nn
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.helpers.log import log


class TsRnnNet(torch.nn.Module):
    def __init__(self,
                 encoder_span: dict,  # contains index span for each encoder
                 target_name: str,
                 input_size: int = None,
                 output_size: int = None,  # forecast horizon length
                 num_hidden: int = None,
                 dropout: float = None
                 ) -> None:

        super().__init__()
        self.target = target_name
        self.encoder_span = encoder_span
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden = num_hidden

        # useless?
        self.dropout = dropout

        # self.ar_column = f'__mdb_ts_previous_{self.target}'
        # linears = [nn.Linear(in_features=inf, out_features=outf) for inf, outf in dims]
        hidden_size = 16
        num_layers = 1
        self.rnn = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self.to(get_devices()[0])

    def to(self, device=None, available_devices=None):
        if 'cuda' not in str(torch.device) == 0:
            log.warning(
                'Creating neural network on CPU, it will be significantly slower than using a GPU, consider using a GPU instead')  # noqa
        self.rnn = self.rnn.to(device)
        self.fc = self.fc.to(device)
        self.device = device
        return self

    def forward(self, input):
        with LightwoodAutocast():
            if len(input.shape) == 1:
                input = input.unsqueeze(0)
            if len(input.shape) == 2:
                input = input.unsqueeze(1)

            fh = torch.zeros((input.shape[0], self.output_size))
            hn = None

            for step in range(self.output_size):
                output, hn = self.rnn(input, hn)
                output = self.fc(output).squeeze()
                fh[:, step] = output

        return fh
