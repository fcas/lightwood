import unittest
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import json_ai_from_problem, predictor_from_json_ai, JsonAI, code_from_json_ai, predictor_from_code  # noqa

np.random.seed(42)


class TestBasic(unittest.TestCase):

    def get_submodels(self):
        submodels = [
            {
                'module': 'XGBoostMixer',
                'args': {
                    'stop_after': '$problem_definition.seconds_per_mixer',
                    'fit_on_dev': True,
                    'target': '$target',
                    'dtype_dict': '$dtype_dict',
                    'target_encoder': '$encoders[self.target]',
                    'use_optuna': True
                }
            },
        ]
        return submodels

    def test_0_basic_regression(self):
        df = pd.read_csv('tests/data/concrete_strength.csv')[:500]
        target = 'concrete_strength'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 80})
        jai = json_ai_from_problem(df, pdef)

        jai.model['args']['submodels'] = self.get_submodels()
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictor.predict(df)

    def test_1_basic_binary(self):
        df = pd.read_csv('tests/data/ionosphere.csv')[:100]
        target = 'target'

        pdef = ProblemDefinition.from_dict({'target': target, 'time_aim': 20, 'unbias_target': False})
        jai = json_ai_from_problem(df, pdef)

        jai.model['args']['submodels'] = [
            {
                'module': 'XGBoostMixer',
                'args': {'stop_after': '$problem_definition.seconds_per_mixer', 'fit_on_dev': True}}
        ]
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(df)
        predictions = predictor.predict(df)

        acc = balanced_accuracy_score(df[target], predictions['prediction'])
        self.assertTrue(acc > 0.5)
        self.assertTrue(all([0 <= p <= 1 for p in predictions['confidence']]))

    @unittest.skipIf(importlib.util.find_spec('ray') is not None, "Ray is available, skipping as this will fail otherwise")
    def test_2_weighted_regression(self):
        """
        This test mocks a dataset intended to demonstrate the efficacy of weighting. The operation does not successfully
        test if the weighting procedure works as intended, but does test the code for bugs.
        """

        # generate data that mocks an observational skew by adding a linear selection to data
        data_size = 100000
        loc = 100.0
        scale = 10.0
        eps = .1
        target_data = np.random.normal(loc=loc, scale=scale, size=data_size)
        epsilon = np.random.normal(loc=0.0, scale=loc * eps, size=len(target_data))
        feature_data = target_data + epsilon
        df = pd.DataFrame({'feature': feature_data, 'target': target_data})

        hist, bin_edges = np.histogram(target_data, bins=10, density=False)
        fracs = np.linspace(1, 100, len(hist))
        fracs = fracs / fracs.sum()
        target_size = 10000
        skewed_arr_list = []
        for i in range(len(hist)):
            frac = fracs[i]
            low_edge = bin_edges[i]
            high_edge = bin_edges[i + 1]

            bin_array = target_data[target_data <= high_edge]
            bin_array = bin_array[bin_array >= low_edge]

            # select only a fraction fo the elements in this bin
            bin_array = bin_array[:int(target_size * frac)]

            skewed_arr_list.append(bin_array)

        skewed_arr = np.concatenate(skewed_arr_list)
        epsilon = np.random.normal(loc=0.0, scale=loc * eps, size=len(skewed_arr))
        skewed_feat = skewed_arr + epsilon
        skew_df = pd.DataFrame({'feature': skewed_feat, 'target': skewed_arr})

        # generate data set weights to remove bias.
        hist, bin_edges = np.histogram(skew_df['target'].to_numpy(), bins=10, density=False)
        hist = 1 - hist / hist.sum()
        target_weights = {bin_edge: bin_frac for bin_edge, bin_frac in zip(bin_edges, hist)}

        pdef = ProblemDefinition.from_dict({'target': 'target', 'target_weights': target_weights, 'time_aim': 80})
        jai = json_ai_from_problem(skew_df, pdef)

        jai.model['args']['submodels'] = self.get_submodels()
        code = code_from_json_ai(jai)
        predictor = predictor_from_code(code)

        predictor.learn(skew_df)
        output_df = predictor.predict(df)

        output_mean = output_df['prediction'].mean()

        self.assertTrue(np.all(np.isclose(output_mean, loc, atol=0., rtol=.03)),
                        msg=f"the output mean {output_mean} is not close to {loc}")
