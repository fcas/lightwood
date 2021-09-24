# TODO: lookup_encoder is awkward; similar to dtype, can we make a file with encoder_lookup? People may be interested
# in seeing where these come from and it's not clear that you need to look here.
# TODO: What does `target_class_distribution` and `positive_domain` do?
# TODO: generate_json_ai is really large; can we abstract it into smaller functions to make it more readable?
# TODO: add_implicit_values unit test ensures NO changes for a fully specified file.
# TODO: Please fix spelling on parallel_preped_encoders
import lightwood.encoder
from typing import Dict
from lightwood.helpers.templating import call, inline_dict, align
import black
from lightwood.api import dtype
import numpy as np
from lightwood.api.types import (
    JsonAI,
    TypeInformation,
    StatisticalAnalysis,
    Feature,
    Output,
    ProblemDefinition,
)

IMPORT_EXTERNAL_DIRS = """
for import_dir in [os.path.expanduser('~/lightwood_modules'), '/etc/lightwood_modules']:
    if os.path.exists(import_dir) and os.access(import_dir, os.R_OK):
        for file_name in list(os.walk(import_dir))[0][2]:
            mod_name = file_name.rstrip('.py')
            loader = importlib.machinery.SourceFileLoader(mod_name,
                                                          os.path.join(import_dir, file_name))
            module = ModuleType(loader.name)
            loader.exec_module(module)
            exec(f'{mod_name} = module')
"""
IMPORTS = """
import lightwood
from lightwood.analysis import *
from lightwood.api import *
from lightwood.data import *
from lightwood.encoder import *
from lightwood.ensemble import *
from lightwood.helpers.device import *
from lightwood.helpers.general import *
from lightwood.helpers.log import *
from lightwood.helpers.numeric import *
from lightwood.helpers.parallelism import *
from lightwood.helpers.seed import *
from lightwood.helpers.text import *
from lightwood.helpers.torch import *
from lightwood.mixer import *
import pandas as pd
from typing import Dict, List
import os
import importlib.machinery
from types import ModuleType
import sys"""


def lookup_encoder(
    col_dtype: str,
    col_name: str,
    is_target: bool,
    problem_defintion: ProblemDefinition,
    is_target_predicting_encoder: bool,
):
    """
    Assign a default encoder for a given column based on its data type, and whether it is a target. Encoders intake raw (but cleaned) data and return an feature representation. This function assigns, per data type, what the featurizer should be. This function runs on each column within the dataset available for model building to assign how it should be featurized.

    Users may override to create a custom encoder to enable their own featurization process. However, in order to generate a template JSON-AI, this code is run first. Users may edit the generated syntax and use custom approaches while model building.

    For each encoder, "args" may be passed. These args depend an encoder requires during its preparation call.

    :param col_dtype: A data-type of a column specified
    :param col_name: The name of the column
    :param is_target: Whether the column is the target for prediction. If true, only certain possible feature representations are allowed, particularly for complex data types.
    :param problem_definition: The ``ProblemDefinition`` criteria; this populates specifics on how models and encoders may be trained.
    :param is_target_predicting_encoder:
    """ # noqa
    exec(IMPORTS)
    exec(IMPORT_EXTERNAL_DIRS)
    tss = problem_defintion.timeseries_settings
    encoder_lookup = {
        dtype.integer: 'Integer.NumericEncoder',
        dtype.float: 'Float.NumericEncoder',
        dtype.binary: 'Binary.BinaryEncoder',
        dtype.categorical: 'Categorical.CategoricalAutoEncoder',
        dtype.tags: 'Tags.MultiHotEncoder',
        dtype.date: 'Date.DatetimeEncoder',
        dtype.datetime: 'Datetime.DatetimeEncoder',
        dtype.image: 'Image.Img2VecEncoder',
        dtype.rich_text: 'Rich_Text.PretrainedLangEncoder',
        dtype.short_text: 'Short_Text.CategoricalAutoEncoder',
        dtype.array: 'Array.ArrayEncoder',
        dtype.tsarray: 'TimeSeries.TimeSeriesEncoder',
        dtype.quantity: 'Quantity.NumericEncoder',
    }

    # If column is a target, only specific feature representations are allowed that enable supervised tasks
    target_encoder_lookup_override = {
        dtype.rich_text: 'Rich_Text.VocabularyEncoder',
        dtype.categorical: 'Categorical.OneHotEncoder',
    }

    # Assign a default encoder to each column.
    encoder_dict = {"module": encoder_lookup[col_dtype], "args": {}}

    # If the column is a target, ensure that the feature representation can enable supervised tasks
    if is_target:
        encoder_dict['args'] = {'is_target': 'True'}

        if col_dtype in target_encoder_lookup_override:
            encoder_dict['module'] = target_encoder_lookup_override[col_dtype]

        if col_dtype in (dtype.categorical, dtype.binary):
            if problem_defintion.unbias_target:
                encoder_dict['args']['target_class_distribution'] = '$statistical_analysis.target_class_distribution'

        if col_dtype in (dtype.integer, dtype.float, dtype.array, dtype.tsarray):
            encoder_dict['args'][
                "positive_domain"
            ] = "$statistical_analysis.positive_domain"

    # Time-series representations require more advanced flags
    if tss.is_timeseries:
        gby = tss.group_by if tss.group_by is not None else []
        if col_name in tss.order_by + tss.historical_columns:
            encoder_dict['module'] = col_dtype.capitalize() + ".TimeSeriesEncoder"
            encoder_dict['args']['original_type'] = f'"{col_dtype}"'
            encoder_dict['args']['target'] = "self.target"
            encoder_dict['args']['grouped_by'] = f"{gby}"

        if is_target:
            if col_dtype in [dtype.integer]:
                encoder_dict['args']['grouped_by'] = f"{gby}"
                encoder_dict['module'] = "Integer.TsNumericEncoder"
            if col_dtype in [dtype.float]:
                encoder_dict['args']['grouped_by'] = f"{gby}"
                encoder_dict['module'] = "Float.TsNumericEncoder"
            if tss.nr_predictions > 1:
                encoder_dict['args']['grouped_by'] = f"{gby}"
                encoder_dict['args']['timesteps'] = f"{tss.nr_predictions}"
                encoder_dict['module'] = 'TimeSeries.TsArrayNumericEncoder'
        if '__mdb_ts_previous' in col_name:
            encoder_dict['module'] = 'Array.ArrayEncoder'
            encoder_dict['args']['original_type'] = f'"{tss.target_type}"'
            encoder_dict['args']['window'] = f'{tss.window}'

    # Set arguments for the encoder
    if encoder_dict['module'] == "Rich_Text.PretrainedLangEncoder" and not is_target:
        encoder_dict['args']['output_type'] = "$dtype_dict[$target]"

    if eval(encoder_dict['module'].split(".")[1]).is_trainable_encoder:
        encoder_dict['args'][
            "stop_after"
        ] = "$problem_definition.seconds_per_encoder"

    if is_target_predicting_encoder:
        encoder_dict['args']['embed_mode'] = 'False'
    return encoder_dict


def generate_json_ai(
    type_information: TypeInformation,
    statistical_analysis: StatisticalAnalysis,
    problem_definition: ProblemDefinition,
) -> JsonAI:
    """
    Given ``TypeInformation``, ``StatisticalAnalysis``, and the ``ProblemDefinition``, generate a JSON config file with the necessary elements of the ML pipeline populated.

    :param TypeInformation: Specifies what data types each column within the dataset are
    :param statistical_analysis:
    :param problem_definition: Specifies details of the model training/building procedure, as defined by ``ProblemDefinition``

    :returns: JSON-AI object with fully populated details of the ML pipeline
    """ # noqa
    exec(IMPORTS)
    exec(IMPORT_EXTERNAL_DIRS)
    target = problem_definition.target
    input_cols = []
    for col_name, col_dtype in type_information.dtypes.items():
        if (
            col_name not in type_information.identifiers
            and col_dtype not in (dtype.invalid, dtype.empty)
            and col_name != target
        ):
            input_cols.append(col_name)

    tss = problem_definition.timeseries_settings
    is_target_predicting_encoder = False
    # Single text column classification
    if (
        len(input_cols) == 1
        and type_information.dtypes[input_cols[0]] in (dtype.rich_text)
        and type_information.dtypes[target] in (dtype.categorical, dtype.binary)
    ):
        is_target_predicting_encoder = True

    if is_target_predicting_encoder:
        mixers = [{
            'module': 'Unit',
            'args': {
                'target_encoder': '$encoders[self.target]',
                'stop_after': '$problem_definition.seconds_per_mixer'
            }
        }]
    else:
        mixers = [{
            'module': 'Neural',
            'args': {
                'fit_on_dev': True,
                'stop_after': '$problem_definition.seconds_per_mixer',
                'search_hyperparameters': True
            }

        }]

        if not tss.is_timeseries or tss.nr_predictions == 1:
            mixers.extend([{
                'module': 'LightGBM',
                'args': {
                    'stop_after': '$problem_definition.seconds_per_mixer',
                    'fit_on_dev': True
                }
            },
                {
                    'module': 'Regression',
                    'args': {
                        'stop_after': '$problem_definition.seconds_per_mixer',
                    }
            }
            ])
        elif tss.nr_predictions > 1:
            mixers.extend([{
                'module': 'LightGBMArray',
                'args': {
                    'fit_on_dev': True,
                    'stop_after': '$problem_definition.seconds_per_mixer',
                    'n_ts_predictions': '$problem_definition.timeseries_settings.nr_predictions'
                }
            }])

            if tss.use_previous_target:
                mixers.extend([
                    {
                        'module': 'SkTime',
                        'args': {
                            'stop_after': '$problem_definition.seconds_per_mixer',
                            'n_ts_predictions': '$problem_definition.timeseries_settings.nr_predictions',
                        },
                    }
                ])

    outputs = {target: Output(
        data_dtype=type_information.dtypes[target],
        encoder=None,
        mixers=mixers,
        ensemble={
            'module': 'BestOf',
            'args': {
                'accuracy_functions': '$accuracy_functions',
            }
        }
    )}

    if (
        tss.is_timeseries and tss.nr_predictions > 1
    ):
        list(outputs.values())[0].data_dtype = dtype.tsarray

    list(outputs.values())[0].encoder = lookup_encoder(
        type_information.dtypes[target], target, True, problem_definition, False
    )

    features: Dict[str, Feature] = {}
    for col_name in input_cols:
        col_dtype = type_information.dtypes[col_name]
        dependency = []
        encoder = lookup_encoder(
            col_dtype, col_name, False, problem_definition, is_target_predicting_encoder
        )

        if tss.is_timeseries and eval(encoder['module'].split(".")[1]).is_timeseries_encoder:
            if tss.group_by is not None:
                for group in tss.group_by:
                    dependency.append(group)

            if tss.use_previous_target:
                dependency.append(f"__mdb_ts_previous_{target}")

        if len(dependency) > 0:
            feature = Feature(encoder=encoder, dependency=dependency)
        else:
            feature = Feature(encoder=encoder)
        features[col_name] = feature

    # Decide on the accuracy functions to use
    output_dtype = list(outputs.values())[0].data_dtype
    if output_dtype in [dtype.integer, dtype.float, dtype.date, dtype.datetime]:
        accuracy_functions = ['r2_score']
    elif output_dtype in [dtype.categorical, dtype.tags, dtype.binary]:
        accuracy_functions = ['balanced_accuracy_score']
    elif output_dtype in (dtype.array, dtype.tsarray):
        accuracy_functions = ['evaluate_array_accuracy']
    else:
        raise Exception(f'Please specify a custom accuracy function for output type {output_dtype}')

    if problem_definition.time_aim is None and (
            problem_definition.seconds_per_mixer is None or problem_definition.seconds_per_encoder is None):
        problem_definition.time_aim = 1000 + np.log(
            statistical_analysis.nr_rows / 10 + 1) * np.sum(
            [4
             if x in [dtype.rich_text, dtype.short_text, dtype.array,
                      dtype.tsarray, dtype.video, dtype.audio, dtype.image]
             else 1
             for x in type_information.dtypes.values()]) * 200

    if problem_definition.time_aim is not None:
        nr_trainable_encoders = len([x for x in features.values() if
                                    eval(x.encoder['module'].split('.')[1]).is_trainable_encoder])
        nr_mixers = len(list(outputs.values())[0].mixers)
        encoder_time_budget_pct = max(3.3 / 5, 1.5 + np.log(nr_trainable_encoders + 1) / 5)

        if nr_trainable_encoders == 0:
            problem_definition.seconds_per_encoder = 0
        else:
            problem_definition.seconds_per_encoder = int(
                problem_definition.time_aim * (encoder_time_budget_pct / nr_trainable_encoders))
        problem_definition.seconds_per_mixer = int(
            problem_definition.time_aim * ((1 / encoder_time_budget_pct) / nr_mixers))

    return JsonAI(
        cleaner=None,
        splitter=None,
        analyzer=None,
        explainer=None,
        features=features,
        outputs=outputs,
        problem_definition=problem_definition,
        identifiers=type_information.identifiers,
        timeseries_transformer=None,
        timeseries_analyzer=None,
        accuracy_functions=accuracy_functions,
    )


def populate_implicit_field(json_ai: JsonAI, field_name: str, implicit_value: dict, is_timeseries: bool) -> None:
    """
    Populate the implicit field of the JsonAI, either by filling it in entirely if missing, or by introspecting the class or function and assigning default values to the args in it's signature that are in the implicit default but haven't been populated by the user

    :params: json_ai: ``JsonAI`` object that describes the ML pipeline that may not have every detail fully specified.
    :params: field_name: Name of the field the implicit field in ``JsonAI``
    :params: implicit_value: The dictionary containing implicit values for the module and arg in the field
    :params: is_timeseries: Whether or not this is a timeseries problem

    :returns: nothing, this method mutates the respective field of the ``JsonAI`` object it receives
    """ # noqa
    # These imports might be slow, in which case the only <easy> solution is to line this code
    exec(IMPORTS)
    exec(IMPORT_EXTERNAL_DIRS)

    field = json_ai.__getattribute__(field_name)
    if field is None:
        if is_timeseries or field_name not in ('timeseries_analyzer', 'timeseries_transformer'):
            field = implicit_value
    else:
        args = eval(field['module']).__code__.co_varnames
        for arg in args:
            if 'args' not in field:
                field['args'] = implicit_value['args']
            else:
                if arg not in field['args']:
                    if arg in implicit_value['args']:
                        field['args'][arg] = implicit_value['args'][arg]
    json_ai.__setattr__(field_name, field)

def add_implicit_values(json_ai: JsonAI) -> JsonAI:
    """
    To enable brevity in writing, auto-generate the "unspecified/missing" details required in the ML pipeline.

    :params: json_ai: ``JsonAI`` object that describes the ML pipeline that may not have every detail fully specified.

    :returns: ``JSONAI`` object with all necessary parameters that were previously left unmentioned filled in.
    """
    problem_definition = json_ai.problem_definition
    tss = problem_definition.timeseries_settings

    # Add implicit arguments
    # @TODO: Consider removing once we have a proper editor in studio
    mixers = json_ai.outputs[json_ai.problem_definition.target].mixers
    for i in range(len(mixers)):
        if mixers[i]['module'] == 'Unit':
            pass
        elif mixers[i]['module'] == 'Neural':
            mixers[i]['args']['target_encoder'] = mixers[i]['args'].get('target_encoder', '$encoders[self.target]')
            mixers[i]['args']['target'] = mixers[i]['args'].get('target', '$target')
            mixers[i]['args']['dtype_dict'] = mixers[i]['args'].get('dtype_dict', '$dtype_dict')
            mixers[i]['args']['input_cols'] = mixers[i]['args'].get('input_cols', '$input_cols')
            mixers[i]['args']['timeseries_settings'] = mixers[i]['args'].get(
                'timeseries_settings', '$problem_definition.timeseries_settings')
            mixers[i]['args']['net'] = mixers[i]['args'].get(
                'net', '"DefaultNet"' if not tss.is_timeseries
                or not tss.use_previous_target
                else '"ArNet"')

        elif mixers[i]['module'] == 'LightGBM':
            mixers[i]['args']['target'] = mixers[i]['args'].get('target', '$target')
            mixers[i]['args']['dtype_dict'] = mixers[i]['args'].get('dtype_dict', '$dtype_dict')
            mixers[i]['args']['input_cols'] = mixers[i]['args'].get('input_cols', '$input_cols')
        elif mixers[i]['module'] == 'Regression':
            mixers[i]['args']['target'] = mixers[i]['args'].get('target', '$target')
            mixers[i]['args']['dtype_dict'] = mixers[i]['args'].get('dtype_dict', '$dtype_dict')
            mixers[i]['args']['target_encoder'] = mixers[i]['args'].get('target_encoder', '$encoders[self.target]')
        elif mixers[i]['module'] == 'LightGBMArray':
            mixers[i]['args']['target'] = mixers[i]['args'].get('target', '$target')
            mixers[i]['args']['dtype_dict'] = mixers[i]['args'].get('dtype_dict', '$dtype_dict')
            mixers[i]['args']['input_cols'] = mixers[i]['args'].get('input_cols', '$input_cols')
        elif mixers[i]['module'] == 'SkTime':
            mixers[i]['args']['target'] = mixers[i]['args'].get('target', '$target')
            mixers[i]['args']['dtype_dict'] = mixers[i]['args'].get('dtype_dict', '$dtype_dict')
            mixers[i]['args']['ts_analysis'] = mixers[i]['args'].get('ts_analysis', '$ts_analysis')

    ensemble = json_ai.outputs[json_ai.problem_definition.target].ensemble
    ensemble['args']['target'] = ensemble['args'].get('target', '$target')
    ensemble['args']['data'] = ensemble['args'].get('data', 'test_data')
    ensemble['args']['mixers'] = ensemble['args'].get('mixers', '$mixers')

    for name in json_ai.features:
        if json_ai.features[name].dependency is None:
            json_ai.features[name].dependency = []
        if json_ai.features[name].data_dtype is None:
            json_ai.features[name].data_dtype = (
                json_ai.features[name].encoder['module'].split(".")[0].lower()
            )

    # Add "hidden" fields
    hidden_fields = [('cleaner', {
        "module": "cleaner",
        "args": {
            "pct_invalid": "$problem_definition.pct_invalid",
            "ignore_features": "$problem_definition.ignore_features",
            "identifiers": "$identifiers",
            "data": "data",
            "dtype_dict": "$dtype_dict",
            "target": "$target",
            "mode": "$mode",
            "timeseries_settings": "$problem_definition.timeseries_settings",
            "anomaly_detection": "$problem_definition.anomaly_detection",
        },
    }), ('splitter', {
        'module': 'splitter',
        'args': {
            'tss': '$problem_definition.timeseries_settings',
            'data': 'data',
            'k': 'nsubsets'
        }
    }), ('analyzer', {
         "module": "model_analyzer",
         "args": {
             "stats_info": "$statistical_analysis",
             "ts_cfg": "$problem_definition.timeseries_settings",
             "accuracy_functions": "$accuracy_functions",
             "predictor": "$ensemble",
             "data": "test_data",
             "train_data": "train_data",
             "target": "$target",
             "disable_column_importance": "False",
             "dtype_dict": "$dtype_dict",
             "fixed_significance": None,
             "confidence_normalizer": False,
             "positive_domain": "$statistical_analysis.positive_domain",
         },
         }), ('explainer', {
             "module": "explain",
             "args": {
                 "timeseries_settings": "$problem_definition.timeseries_settings",
                 "positive_domain": "$statistical_analysis.positive_domain",
                 "fixed_confidence": "$problem_definition.fixed_confidence",
                 "anomaly_detection": "$problem_definition.anomaly_detection",
                 "anomaly_error_rate": "$problem_definition.anomaly_error_rate",
                 "anomaly_cooldown": "$problem_definition.anomaly_cooldown",
                 "data": "data",
                 "encoded_data": "encoded_data",
                 "predictions": "df",
                 "analysis": "$runtime_analyzer",
                 "ts_analysis": "$ts_analysis" if tss.is_timeseries else None,
                 "target_name": "$target",
                 "target_dtype": "$dtype_dict[self.target]",
             },
         }), ('timeseries_transformer', {
             "module": "transform_timeseries",
             "args": {
                 "timeseries_settings": "$problem_definition.timeseries_settings",
                 "data": "data",
                 "dtype_dict": "$dtype_dict",
                 "target": "$target",
                 "mode": "$mode",
             },
         }), ('timeseries_analyzer', {
             "module": "timeseries_analyzer",
             "args": {
                 "timeseries_settings": "$problem_definition.timeseries_settings",
                 "data": "data",
                 "dtype_dict": "$dtype_dict",
                 "target": "$target",
             },
         })]

    for field_name, implicit_value in hidden_fields:
        populate_implicit_field(json_ai, field_name, implicit_value, tss.is_timeseries)

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    """
    Generates a custom ``PredictorInterface`` given the specifications from ``JsonAI`` object.

    :param json_ai: ``JsonAI`` object with fully specified parameters

    :returns: Automated syntax of the ``PredictorInterface`` object.
    """
    # Fill in any missing values
    json_ai = add_implicit_values(json_ai)

    encoder_dict = {json_ai.problem_definition.target: call(list(json_ai.outputs.values())[0].encoder)}
    dependency_dict = {}
    dtype_dict = {
        json_ai.problem_definition.target: f"""'{list(json_ai.outputs.values())[0].data_dtype}'"""
    }

    for col_name, feature in json_ai.features.items():
        if col_name not in json_ai.problem_definition.ignore_features:
            encoder_dict[col_name] = call(feature.encoder)
            dependency_dict[col_name] = feature.dependency
            dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    # @TODO: Move into json-ai creation function (I think? Maybe? Let's discuss)
    tss = json_ai.problem_definition.timeseries_settings
    if tss.is_timeseries and tss.use_previous_target:
        col_name = f'__mdb_ts_previous_{json_ai.problem_definition.target}'
        json_ai.problem_definition.timeseries_settings.target_type = list(json_ai.outputs.values())[0].data_dtype
        encoder_dict[col_name] = call(lookup_encoder(list(json_ai.outputs.values())[0].data_dtype,
                                                     col_name,
                                                     False,
                                                     json_ai.problem_definition,
                                                     False,
                                                     ))
        dependency_dict[col_name] = []
        dtype_dict[col_name] = f"""'{list(json_ai.outputs.values())[0].data_dtype}'"""
        json_ai.features[col_name] = Feature(encoder=encoder_dict[col_name])

    ignored_cols = json_ai.problem_definition.ignore_features
    input_cols = [x.replace("'", "\\'").replace('"', '\\"')
                  for x in json_ai.features]
    input_cols = ','.join([f"""'{name}'""" for name in input_cols if name not in ignored_cols])

    ts_transform_code = ""
    ts_analyze_code = ""
    ts_encoder_code = ""
    if json_ai.timeseries_transformer is not None:
        ts_transform_code = f"""
log.info('Transforming timeseries data')
data = {call(json_ai.timeseries_transformer)}
"""
        ts_analyze_code = f"""
self.ts_analysis = {call(json_ai.timeseries_analyzer)}
"""

    if json_ai.timeseries_analyzer is not None:
        ts_encoder_code = f"""
if encoder.is_timeseries_encoder:
    kwargs['ts_analysis'] = self.ts_analysis
"""

    if json_ai.problem_definition.timeseries_settings.is_timeseries:
        ts_target_code = """
if encoder.is_target:
    encoder.normalizers = self.ts_analysis['target_normalizers']
    encoder.group_combinations = self.ts_analysis['group_combinations']
"""
    else:
        ts_target_code = ""

    dataprep_body = f"""
# The type of each column
self.problem_definition = ProblemDefinition.from_dict({json_ai.problem_definition.to_dict()})
self.accuracy_functions = {json_ai.accuracy_functions}
self.identifiers = {json_ai.identifiers}
self.dtype_dict = {inline_dict(dtype_dict)}
self.statistical_analysis = lightwood.data.statistical_analysis(data, self.dtype_dict, {json_ai.identifiers},
                                                                self.problem_definition)
self.mode = 'train'
# How columns are encoded
self.encoders = {inline_dict(encoder_dict)}
# Which column depends on which
self.dependencies = {inline_dict(dependency_dict)}
#
self.input_cols = [{input_cols}]

log.info('Cleaning the data')
data = {call(json_ai.cleaner)}

{ts_transform_code}
{ts_analyze_code}

nsubsets = {json_ai.problem_definition.nsubsets}
log.info(f'Splitting the data into {{nsubsets}} subsets')
subsets = {call(json_ai.splitter)}

log.info('Preparing the encoders')

encoder_preping_dict = {{}}
enc_preping_data = pd.concat(subsets[0:nsubsets-1])
for col_name, encoder in self.encoders.items():
    if not encoder.is_nn_encoder:
        encoder_preping_dict[col_name] = [encoder, enc_preping_data[col_name], 'prepare']
        log.info(f'Encoder preping dict length of: {{len(encoder_preping_dict)}}')

parallel_preped_encoders = mut_method_call(encoder_preping_dict)
for col_name, encoder in parallel_preped_encoders.items():
    self.encoders[col_name] = encoder

if self.target not in parallel_preped_encoders:
    self.encoders[self.target].prepare(enc_preping_data[self.target])

for col_name, encoder in self.encoders.items():
    if encoder.is_nn_encoder:
        priming_data = pd.concat(subsets[0:nsubsets-1])
        kwargs = {{}}
        if self.dependencies[col_name]:
            kwargs['dependency_data'] = {{}}
            for col in self.dependencies[col_name]:
                kwargs['dependency_data'][col] = {{
                    'original_type': self.dtype_dict[col],
                    'data': priming_data[col]
                }}
            {align(ts_encoder_code, 3)}

        # This assumes target  encoders are also prepared in parallel, might not be true
        if hasattr(encoder, 'uses_target'):
            kwargs['encoded_target_values'] = parallel_preped_encoders[self.target].encode(priming_data[self.target])

        encoder.prepare(priming_data[col_name], **kwargs)

    {align(ts_target_code, 1)}
"""
    dataprep_body = align(dataprep_body, 2)

    learn_body = f"""
log.info('Featurizing the data')

encoded_ds_arr = lightwood.encode(self.encoders, subsets, self.target)
train_data = encoded_ds_arr[0:int(nsubsets*0.9)]
test_data = encoded_ds_arr[int(nsubsets*0.9):]

log.info('Training the mixers')
self.mixers = [{', '.join([call(x) for x in list(json_ai.outputs.values())[0].mixers])}]
trained_mixers = []
for mixer in self.mixers:
    try:
        mixer.fit(train_data)
        trained_mixers.append(mixer)
    except Exception as e:
        log.warning(f'Exception: {{e}} when training mixer: {{mixer}}')
        if {json_ai.problem_definition.strict_mode} and mixer.stable:
            raise e

self.mixers = trained_mixers

log.info('Ensembling the mixer')
self.ensemble = {call(list(json_ai.outputs.values())[0].ensemble)}
self.supports_proba = self.ensemble.supports_proba

log.info('Analyzing the ensemble')
self.model_analysis, self.runtime_analyzer = {call(json_ai.analyzer)}

# Enable partial fit of model, after its trained, on validation data. This is ONLY to be used in cases where there is
# an expectation of testing data and a continuously evolving pipeline; this assumes that all data available is
# important to train with.
for mixer in self.mixers:
    if {json_ai.problem_definition.fit_on_validation}:
        mixer.partial_fit(test_data, train_data)
"""
    learn_body = align(learn_body, 2)

    predict_common_body = f"""
self.mode = 'predict'
log.info('Cleaning the data')
data = {call(json_ai.cleaner)}

{ts_transform_code}

encoded_ds = lightwood.encode(self.encoders, data, self.target)[0]
encoded_data = encoded_ds.get_encoded_data(include_target=False)
"""
    predict_common_body = align(predict_common_body, 2)

    predict_body = f"""
df = self.ensemble(encoded_ds)
insights = {call(json_ai.explainer)}
return insights
"""
    predict_body = align(predict_body, 2)

    predict_proba_body = f"""
df = self.ensemble(encoded_ds, predict_proba=True)
insights = {call(json_ai.explainer)}
return insights
"""
    predict_proba_body = align(predict_proba_body, 2)

    predictor_code = f"""
{IMPORTS}
{IMPORT_EXTERNAL_DIRS}

class Predictor(PredictorInterface):
    target: str
    mixers: List[BaseMixer]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble
    mode: str

    def __init__(self):
        seed({json_ai.problem_definition.seed_nr})
        self.target = '{json_ai.problem_definition.target}'
        self.mode = 'innactive'

    def learn(self, data: pd.DataFrame) -> None:
{dataprep_body}
{learn_body}

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
{predict_common_body}
{predict_body}


    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
{predict_common_body}
{predict_proba_body}
"""

    predictor_code = black.format_str(predictor_code, mode=black.FileMode())

    return predictor_code


def validate_json_ai(json_ai: JsonAI) -> bool:
    """
    Checks the validity of a ``JsonAI`` object
    
    :param json_ai: A ``JsonAI`` object

    :returns: Wether the JsonAI is valid, i.e. doesn't contain prohibited values, unknown values and can be turned into code.
    """ # noqa
    from lightwood.api.high_level import predictor_from_code, code_from_json_ai

    try:
        predictor_from_code(code_from_json_ai(json_ai))
        return True
    except Exception:
        return False
