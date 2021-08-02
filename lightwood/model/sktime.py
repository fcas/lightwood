import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster

from lightwood.api import dtype
from lightwood.helpers.log import log
from lightwood.model.base import BaseModel
from lightwood.encoder.time_series.helpers.common import get_group_matches
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class SkTime(BaseModel):
    forecaster: str
    n_ts_predictions: int
    target: str
    supports_proba: bool

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], n_ts_predictions: int, ts_analysis: Dict):
        super().__init__(stop_after)
        self.target = target
        dtype_dict[target] = dtype.float
        self.model_class = AutoARIMA
        self.models = {}
        self.n_ts_predictions = n_ts_predictions
        self.ts_analysis = ts_analysis
        self.forecasting_horizon = np.arange(1, self.n_ts_predictions)
        self.cutoff_index = {}  # marks index at which training data stops and forecasting window starts
        self.grouped_by = ['__default'] if not ts_analysis['tss'].group_by else ts_analysis['tss'].group_by
        self.supports_proba = False
        self.stable = True

    def fit(self, ds_arr: List[EncodedDs]) -> None:
        log.info('Started fitting sktime forecaster for array prediction')

        all_folds = ConcatedEncodedDs(ds_arr)
        df = all_folds.data_frame.sort_values(by=f'__mdb_original_{self.ts_analysis["tss"].order_by[0]}')
        data = {'data': df[self.target],
                'group_info': {gcol: df[gcol].tolist()
                               for gcol in self.grouped_by} if self.ts_analysis['tss'].group_by else {}}

        for group in self.ts_analysis['group_combinations']:
            self.models[group] = self.model_class()

            if self.grouped_by == ['__default']:
                series_idxs = data['data'].index
                series_data = data['data'].values
            else:
                series_idxs, series_data = get_group_matches(data, group)

            if series_data.size > 0:
                series = pd.Series(series_data.squeeze(), index=series_idxs)
                series = series.sort_index(ascending=True)
                series = series.reset_index(drop=True)
                try:
                    self.models[group].fit(series)
                except ValueError:
                    self.models[group] = self.model_class(deseasonalize=False)
                    self.models[group].fit(series)

                self.cutoff_index[group] = len(series)

            if self.grouped_by == ['__default']:
                break

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs], predict_proba: bool = False) -> pd.DataFrame:
        if predict_proba:
            log.warning('This model cannot output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=['prediction'],
                           dtype=object)

        data = {'data': ds.data_frame[self.target].reset_index(drop=True),
                'group_info': {gcol: ds.data_frame[gcol].tolist()
                               for gcol in self.grouped_by} if self.ts_analysis['tss'].group_by else {}}

        # all_idxs = list(range(length))  # @TODO: substract, and assign empty predictions to remainder

        for group in self.ts_analysis['group_combinations']:

            if self.grouped_by == ['__default']:
                series_idxs = data['data'].index
                series_data = data['data'].values
            else:
                series_idxs, series_data = get_group_matches(data, group)

            if series_data.size > 0:
                series = pd.Series(series_data.squeeze(), index=series_idxs)
                series = series.sort_index(ascending=True)
                series = series.reset_index(drop=True)
                cutoff = self.cutoff_index[group]

                for idx, _ in enumerate(series.iteritems()):
                    ydf['prediction'].iloc[series_idxs[idx]] = self.models[group].predict(
                        np.arange(idx,  # +cutoff
                                  idx+self.n_ts_predictions)).tolist()  # +cutoff

            if self.grouped_by == ['__default']:
                break


        return ydf[['prediction']]
