# Recycling circuit breaker mixer
from torch.cuda.amp.grad_scaler import GradScaler
from lightwood.helpers.device import get_devices
from lightwood.mixer import BaseMixer
from lightwood.encoder import BaseEncoder
from lightwood.data.encoded_ds import EncodedDs
from typing import Dict, List
from lightwood.helpers.torch import LightwoodAutocast
from torch import nn
import torch
import pandas as pd
from lightwood.api.types import PredictionArguments
from lightwood.mixer.helpers.transform_corss_entropy_loss import TransformCrossEntropyLoss
import torch_optimizer as ad_optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import MSELoss
from lightwood.api.dtype import dtype
from lightwood.helpers.general import DummyContextManager


class RCBNet(nn.Module):
    no_loops: int
    null_output: torch.Tensor
    device: torch.device
    blocks: nn.ModuleList
    input_size: int
    start_grad: int

    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(RCBNet, self).__init__()
        self.no_loops = 5
        self.null_output = torch.zeros(output_size)
        self.input_size = input_size
        self.start_grad = 0
        blocks = []
        for idx in reversed(list(range(1, self.no_loops + 1))):
            layer_out_size = int(idx * input_size / self.no_loops + output_size + 1)
            layers = [nn.Linear(layer_out_size - 1, layer_out_size)]
            if idx < self.no_loops:
                layers.append(nn.SELU())

            blocks.append(torch.nn.Sequential(*layers))

        self.blocks = nn.ModuleList(reversed(blocks))
        self.to(device)

    def to(self, device: torch.device) -> torch.nn.Module:
        self.blocks = self.blocks.to(device)
        self.device = device
        return self

    def forward(self, X: torch.Tensor):
        for n in range(1, self.no_loops + 1):
            Xr = None
            Yh = self.null_output.repeat(X.size(0), 1)

            for i in range(n):
                with DummyContextManager() if self.start_grad <= n else torch.no_grad():
                    start_in = int(self.input_size / self.no_loops) * i
                    end_in = int(self.input_size / self.no_loops) * (i + 1)
                    external_input = X[:, start_in:end_in]
                    if Xr is None:
                        Xr = external_input
                    else:
                        Xr = torch.cat([Xr, external_input], 1)

                    Xi = torch.cat([Xr, Yh], 1)
                    Xi = self.blocks[i](Xi)

                    Xr = Xi[:, :end_in]
                    Yh = Xi[:, end_in:-1]
                    breaker = Xi[:, -1].mean()
                    # Maybe in the future do this only for <certain> examples (?)
                    if breaker > 0.9:
                        return Yh
            return Yh


class RCB(BaseMixer):
    model: nn.Module
    dtype_dict: dict
    target: str
    stable: bool = True

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], target_encoder: BaseEncoder,
                 fit_on_dev: bool):
        super().__init__(stop_after)
        self.dtype_dict = dtype_dict
        self.target = target
        self.target_encoder = target_encoder
        self.fit_on_dev = fit_on_dev

    def _select_criterion(self) -> torch.nn.Module:
        if self.dtype_dict[self.target] in (dtype.categorical, dtype.binary):
            criterion = TransformCrossEntropyLoss(weight=self.target_encoder.index_weights.to(self.device))
        elif self.dtype_dict[self.target] in (dtype.tags):
            criterion = nn.BCEWithLogitsLoss()
        elif (self.dtype_dict[self.target] in (dtype.integer, dtype.float, dtype.tsarray, dtype.quantity)
                and self.timeseries_settings.is_timeseries):
            criterion = nn.L1Loss()
        elif self.dtype_dict[self.target] in (dtype.integer, dtype.float, dtype.quantity):
            criterion = MSELoss()
        else:
            criterion = MSELoss()

        return criterion

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        self.device, _ = get_devices()
        self.net = RCBNet(len(train_data[0][0]), len(train_data[0][1]), self.device)

        self.batch_size = min(200, int(len(train_data) / 10))
        self.batch_size = max(40, self.batch_size)

        criterion = self._select_criterion()
        optimizer = ad_optim.Ranger(self.net.parameters(), lr=0.01)
        scaler = GradScaler()

        # dev_dl = DataLoader(dev_data, batch_size=self.batch_size, shuffle=False)
        train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        for i in range(100):
            for X, Y in train_dl:
                with LightwoodAutocast():
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    optimizer.zero_grad()
                    Yh = self.net(X)
                    loss = criterion(Yh, Y)
                    if LightwoodAutocast.active:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    print(loss.item())

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        # @TODO Implement once we get fit to get good results and actually introduce the encoder
        pass

    def __call__(self, ds: EncodedDs, args: PredictionArguments) -> pd.DataFrame:
        self.net = self.net.eval()
        decoded_predictions: List[object] = []

        with torch.no_grad():
            for idx, (X, Y) in enumerate(ds):
                X = X.to(self.net.device)
                Yh = self.net(X)
                Yh = torch.unsqueeze(Yh, 0) if len(Yh.shape) < 2 else Yh

                kwargs = {}
                for dep in self.target_encoder.dependencies:
                    kwargs['dependency_data'] = {dep: ds.data_frame.iloc[idx][[dep]].values}

                decoded_prediction = self.target_encoder.decode(Yh, **kwargs)

                if not self.timeseries_settings.is_timeseries or self.timeseries_settings.nr_predictions == 1:
                    decoded_predictions.extend(decoded_prediction)
                else:
                    decoded_predictions.append(decoded_prediction)

            ydf = pd.DataFrame({'prediction': decoded_predictions})

            return ydf
