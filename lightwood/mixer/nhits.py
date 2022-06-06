from typing import Dict, Union

import numpy as np
import pandas as pd
from hyperopt import hp
import neuralforecast as nf
from neuralforecast.models.mqnhits.mqnhits import MQNHITS

from lightwood.helpers.log import log
from lightwood.mixer.base import BaseMixer
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class NHitsMixer(BaseMixer):
    horizon: int
    target: str
    supports_proba: bool
    model_path: str
    hyperparam_search: bool
    default_config: dict

    def __init__(
            self,
            stop_after: float,
            target: str,
            horizon: int,
            ts_analysis: Dict,
    ):
        """
        Mixer description here.
        
        :param stop_after: time budget in seconds.
        :param target: column to forecast.
        :param horizon: length of forecasted horizon.
        :param ts_analysis: dictionary with miscellaneous time series info, as generated by 'lightwood.data.timeseries_analyzer'.
        """  # noqa
        super().__init__(stop_after)
        self.stable = True
        self.prepared = False
        self.supports_proba = False
        self.target = target
        self.horizon = horizon
        self.ts_analysis = ts_analysis
        self.grouped_by = ['__default'] if not ts_analysis['tss'].group_by else ts_analysis['tss'].group_by

        # pretraining info
        self.pretrained = False  # True  # todo: modifiable from JsonAI, plus option to finetune!
        self.base_url = 'https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/'
        self.model_names = {
            'hourly': 'nhits_m4_hourly.ckpt',  # hourly (non-tiny)
            'daily': 'nhits_m4_daily.ckpt',   # daily
            'monthly': 'nhits_m4_monthyl.ckpt',  # monthly
            'yearly': 'nhits_m4_hourly.ckpt',  # yearly
        }
        self.model = None
        self.config = nf.auto.mqnhits_space(self.horizon)
        self.config['n_time_in'] = self.ts_analysis['tss'].window
        self.config['n_time_out'] = self.horizon
        self.config['n_x_hidden'] = 0
        self.config['n_s_hidden'] = 0
        self.config['frequency'] = self.ts_analysis['sample_freqs']['__default']

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the N-HITS model.
        """  # noqa
        log.info('Started fitting N-HITS forecasting model')

        cat_ds = ConcatedEncodedDs([train_data, dev_data])
        oby_col = self.ts_analysis["tss"].order_by[0]
        df = cat_ds.data_frame.sort_values(by=f'__mdb_original_{oby_col}')

        # 2. adapt data into the expected DFs
        Y_df = self._make_initial_df(df)

        # set val-test cutoff
        n_time = len(df[f'__mdb_original_{oby_col}'].unique())
        n_ts_val = int(.1 * n_time)
        n_ts_test = int(.1 * n_time)

        # 3. TODO: merge user-defined config into default

        # train the model
        n_time_out = self.horizon
        if self.pretrained:
            model_name = self.model_names.get(self.ts_analysis['sample_freqs']['__default'], None)
            model_name = self.model_names['hourly'] if model_name is None else model_name
            ckpt_url = self.base_url + model_name
            self.model = MQNHITS.load_from_checkpoint(ckpt_url)  # TODO use this when not pretraining for consistency

            # TODO: if self.finetune: ...
        else:
            self.model = nf.auto.MQNHITS(horizon=n_time_out)
            self.model.space['max_steps'] = hp.choice('max_steps', [5e4])
            self.model.space['max_epochs'] = hp.choice('max_epochs', [50])
            # self.model.space['max_epochs'] = hp.choice('max_epochs', [-1, 10])
            # 4 & 4 works... theory is both need to be smaller than length of val and/or test idxs... 8&8 fails?
            self.model.space['n_time_in'] = hp.choice('n_time_in', [self.ts_analysis['tss'].window - 1])
            self.model.space['n_time_out'] = hp.choice('n_time_out', [self.horizon])
            self.model.space['n_windows'] = hp.choice('n_windows', [1])
            self.model.fit(Y_df=Y_df,
                           X_df=None,       # Exogenous variables
                           S_df=None,       # Static variables
                           hyperopt_steps=5,
                           n_ts_val=n_ts_val,
                           n_ts_test=n_ts_test,
                           results_dir='./results/autonhits',  # TODO: rm/change to /tmp/lightwood/autonhits or similar
                           save_trials=False,
                           loss_function_val=nf.losses.numpy.mqloss,
                           loss_functions_test={'MQ': nf.losses.numpy.mqloss},
                           return_test_forecast=False,
                           verbose=True)  # False

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Due to how lightwood implements the `update` procedure, expected inputs for this method are:
        
        :param dev_data: original `test` split (used to validate and select model if ensemble is `BestOf`).
        :param train_data: concatenated original `train` and `dev` splits.
        """  # noqa
        self.hyperparam_search = False
        self.fit(dev_data, train_data)
        self.prepared = True

    def __call__(self, ds: Union[EncodedDs, ConcatedEncodedDs],
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Calls the mixer to emit forecasts.
        """  # noqa
        if args.predict_proba:
            log.warning('This mixer does not output probability estimates')

        length = sum(ds.encoded_ds_lenghts) if isinstance(ds, ConcatedEncodedDs) else len(ds)
        ydf = pd.DataFrame(0,  # zero-filled
                           index=np.arange(length),
                           columns=['prediction'],
                           dtype=object)

        input_df = self._make_initial_df(ds.data_frame)  # TODO make it so that it's horizon worth of data in each row
        pred_col = 'y_50'  # median == point prediction
        for i in range(input_df.shape[0]):
            fcst = self.model.forecast(Y_df=input_df.iloc[i:i + 1])
            ydf.iloc[i]['prediction'] = fcst[pred_col].tolist()[:self.horizon]

        return ydf[['prediction']]

    def _make_initial_df(self, df):
        oby_col = self.ts_analysis["tss"].order_by[0]
        Y_df = pd.DataFrame()
        Y_df['y'] = df[self.target]
        Y_df['ds'] = pd.to_datetime(df[f'__mdb_original_{oby_col}'], unit='s')
        if self.grouped_by != ['__default']:
            Y_df['unique_id'] = df[self.grouped_by].apply(lambda x: ','.join([elt for elt in x]), axis=1)
        else:
            Y_df['unique_id'] = ''
        return Y_df
