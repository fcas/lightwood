from lightwood.api.types import ModelAnalysis
import dill
import pandas as pd
from typing import Dict

# Interface that must be respected by predictor objects generated from JSON ML and/or compatible with Mindsdb
class PredictorInterface:
    """
    Abstraction of a Lightwood predictor. The PredictorInterface encompasses how Lightwood interacts with the full ML pipeline. Internally,

    The ``PredictorInterface`` class must have 5 expected functions:

    - ``learn``: An end-to-end technique specifying how to pre-process, featurize, and train the model(s) of interest. The expected input is raw, untrained data. No explicit output is provided, but the Predictor object will "host" the trained model thus.
    - ``adjust``: The manner to incorporate new data to update pre-existing model(s).
    - ``predict``: Deploys the chosen best model, and evaluates the given data to provide target estimates.
    - ``predict_proba``: Deploys the chosen best model, and enables user to analyze how the model makes estimates. This depends on whether the models internally have "predict_proba" as a possible method (thus, only for classification).
    - ``save``: Saves the Predictor object for further use.

    The ``PredictorInterface`` is created via J{ai}son's custom code creation. A problem inherits from this class with pre-populated routines to fill out expected results, given the nature of each problem type.
    """ # noqa

    model_analysis: ModelAnalysis = None

    def __init__(self):
        pass

    def analyze_data(self, data: pd.DataFrame) -> None:
        """
        Performs a statistical analysis on the data to identify distributions, imbalanced classes, and other nuances within the data.

        :param data: Data used in training the model(s).
        """ # noqa
        pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the unprocessed dataset provided.

        :param data: (Unprocessed) Data used in training the model(s).
        :returns: The cleaned data frame
        """ # noqa
        pass

    def split(self, clean_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Categorizes the data into a training/testing split; if data is a classification problem, will stratify the data.

        :param clean_data: Pre-processed data.
        :returns: Dictionary containing training/testing fraction
        """ # noqa
        pass

    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Prepares the encoders for each column of data. 

        :param data: Pre-processed data that has been split into train/test. Explicitly uses "train" and/or "dev" in preparation of encoders.

        :returns: Nothing; prepares the encoders for learned representations.
        """

    def featurize(self, data: Dict[str, pd.DataFrame]):
        """
        Provides an encoded representation for each dataset in ``data``. Requires `self.encoders` to be prepared.

        :param data: Pre-processed data from the dataset, split into train/test (or any other keys relevant)

        :returns: For each dataset provided in ``data``, the encoded representations of the data.
        """ # noqa
        pass

    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        """
        Fits "mixer" models to train predictors on the featurized data. Instantiates a set of trained mixers and an ensemble of them.

        :param enc_data: Pre-processed and featurized data, split into the relevant train/test splits.
        """
        pass

    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        """
        Evaluate the quality of mixers within an ensemble of models.

        :param enc_data: Pre-processed and featurized data, split into the relevant train/test splits.
        """
        pass

    def learn(self, data: pd.DataFrame) -> None:
        """
        Trains the attribute model starting from raw data. Raw data is pre-processed and cleaned accordingly. As data is assigned a particular type (ex: numerical, categorical, etc.), the respective feature encoder will convert it into a representation useable for training ML models. Of all ML models requested, these models are compiled and fit on the training data.

        This step amalgates ``preprocess`` -> ``featurize`` -> ``fit`` with the necessary splitting + analyze_data that occurs. 

        :param data: (Unprocessed) Data used in training the model(s).

        :returns: Nothing; instantiates with best fit model from ensemble.
        """ # noqa
        pass

    def adjust(self, data: pd.DataFrame) -> None:
        """
        Adjusts a previously trained model on new data. Adopts the same process as ``learn`` but with the exception that the `adjust` function expects the best model to have been already trained.

        ..warnings:: Not tested yet - this is an experimental feature
        :param data: New data used to adjust a previously trained model.

        :returns: Nothing; adjusts best-fit model
        """ # noqa
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Intakes raw data to provide predicted values for your trained model.

        :param data: Data (n_samples, n_columns) that the model(s) will evaluate on and provide the target prediction.

        :returns: A dataframe of predictions of the same length of input.
        """ # noqa
        pass

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Intakes raw data to provide some element of confidence/explainability metric to gauge your model's predictive abilities.

        :param data: Data that the model(s) will evaluate on; provides the some element of predictive strength (ex: how "confident" the model is).

        :returns: A dataframe of confidence metrics for each datapoint provided in the input (n_samples, n_classes)
        """ # noqa
        pass

    def save(self, file_path: str) -> None:
        """
        With a provided file path, saves the Predictor instance for later use.

        :param file_path: Location to store your Predictor Instance.

        :returns: Saves Predictor instance.
        """
        with open(file_path, "wb") as fp:
            dill.dump(self, fp)
