from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np


from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.log import log
from lightwood.helpers.parallelism import get_nr_procs

import dill

def dill_dump(my_object,fname):
    with open(f'{fname}.pkl', 'wb') as file:
        dill.dump(my_object, file)
        
    return f'Saved as {fname}'

def dill_load(fname):
    with open(f'{fname}.pkl', 'rb') as file:
        my_object = dill.load(file)
    
    return my_object




class VectorSimilarity(BaseAnalysisBlock):
    """
    
    Xx

    """  # noqa

    def __init__(self, deps: Optional[Tuple] = ...):
        super().__init__(deps=deps)
        self.target = None
        self.input_cols = []
        self.exceptions = ['__make_predictions']
        self.ordinal_encoders = dict()
        
        log.info("Vector Similarity: init")





    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        log.info("Vector Similarity: analyse")
        ns = SimpleNamespace(**kwargs)
        self.target = ns.target
        
        df = deepcopy(ns.train_data.data_frame)
        n_jobs = get_nr_procs(df)

        #clf = SUOD(base_estimators=detector_list,
        #           contamination=self.contamination,
        #           n_jobs=n_jobs,
        #           combination='average',
        #           verbose=False)
        info['train_data'] = df

        if False:
            if ns.tss.is_timeseries:
                for gcol in ns.tss.group_by:
                    self.ordinal_encoders[gcol] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    self.ordinal_encoders[gcol].fit(df[gcol].values.reshape(-1, 1))
                for i in range(1, ns.tss.horizon):
                    self.exceptions.append(f'{self.target}_timestep_{i}')

                self.exceptions.append(ns.tss.order_by)
                df = self._preprocess_ts_df(df, ns)

            for col in df.columns:
                if col != self.target and '__mdb' not in col and col not in self.exceptions:
                    self.input_cols.append(col)

            df = df.loc[:, self.input_cols].dropna()
            clf.fit(df.values)
            info['pyod_explainer'] = clf

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object],
                **kwargs
                ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        
        log.warning("VECTOR SIMILARITY")

        ns = SimpleNamespace(**kwargs)

        df = deepcopy(ns.data)
        df.to_csv('df.csv')
        dill_dump(df,'df')
        log.warning("OUTPUT df")

        df_train = deepcopy(ns.analysis['train_data'])
        df_train.to_csv('df_train.csv')
        dill_dump(df_train,'df_train')
        log.warning("OUTPUT df_train")

            
        return row_insights, global_insights
