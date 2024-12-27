from pycaret.classification import *
import pandas as pd


class ClassicationModel:
    '''Simplify building, training, and managing machine learning classification models using PyCaret.'''

    def __init__(self, training_data: pd.DataFrame, test_data: pd.DataFrame | None = None, target='Class'):
        '''Constructor of classification model.
        Arguments:
        - training_data (pd.DataFrame): The dataset used to train the model.
        - test_data (pd.DataFrame, optional): An optional test dataset for evaluation.
        - target (str): The target column in the dataset for classification. Defaults to 'Class'.

        Attributes:
        - training_data (pd.DataFrame): Stores the training data.
        - test_data (pd.DataFrame): Stores the test data if provided.
        - target (str): Specifies the target column for prediction.
        - model (Any): Stores the trained PyCaret model.
        - _istrained (bool): Tracks whether the model has been trained.
        '''
        self.training_data = training_data
        self.test_data = test_data
        self.target = target
        self.model = None
        self._istrained = False

    def _train_model(self, session_id: int = 123, n_folds: int = 3, model: str | None = None, **kwargs) -> None:
        '''Train machine learning model using PyCaret's setup and model comparison/creation utilities.

        Arguments:
        - session_id (int): Seed for reproducibility. Defaults to 123.
        - n_folds (int): Number of cross-validation folds. Defaults to 3.
        - model (str, optional): The specific model to train. If None, the best model is selected automatically using compare_models.
        - kwargs: Additional keyword arguments passed to setup.

        Attributes Updated:
        - model: Stores the trained model.
        - _istrained: Set to True after training.

        Returns:
        - None
        '''
        clf = setup(data=self.training_data, target=self.target,
                     session_id=session_id, 
                     fix_imbalance = True, 
                     normalize = True,
                     normalize_method='minmax',
                     use_gpu=True,
                     **kwargs)
        if model is None:
            best_model = compare_models(fold = n_folds)
        else:
            best_model = create_model(model, fold = n_folds)
        self.model = best_model
        self._istrained = True

    def _predict(self, data: pd.DataFrame) -> pd.DataFrame | None:
        '''Use the trained model to make predictions on a given dataset.

        Arguments:
        - data (pd.DataFrame): The dataset for which predictions are to be made.

        Returns:
        - pd.DataFrame | None: Prediction results if the model is trained. None if the model is not trained.
        '''
        if self.model is not None:
            return predict_model(self.model,data=data)
        else:
            return None
    
    def _make_dashboard(self) -> None:
        '''Generate an interactive dashboard for the trained model.

        Arguments:
        - None

        Attributes Accessed:
        - model: Used to generate the dashboard.

        Returns:
        - None
        '''
        if self.model is not None:
            dashboard(self.model)
        else:
            return None
        
    def _save_model(self, model_path : str = './models') -> None:
        '''Save the trained model to the specified file path.

        Arguments:
        - model_path (str): The directory or file path where the model should be saved. Defaults to './models'.

        Returns:
        - None
        '''
        save_model(self.model, model_path)
        
    def _load_model(self, model_path:  str = './models') -> None:
        '''Load a saved model from the specified file path.

        Arguments:
        - model_path (str): The directory or file path from which the model should be loaded. Defaults to './models'.

        Attributes Updated:
        - model: Stores the loaded model.
        - _istrained: Set to True after loading the model.

        Returns:
        - None
        '''
        self.model = load_model(model_path)
        self._istrained = True
