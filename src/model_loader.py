import joblib
import shap
import pandas as pd
from src.config import FEATURE_NAMES

def load_model(model_path):
    """
    Load the trained scikit-learn Pipeline model.
    """
    return joblib.load(model_path)


def init_shap_explainer(model):
    """
    Initialize SHAP KernelExplainer for a Pipeline model.

    We wrap model.predict_proba in a function that:
    - Accepts a numpy array
    - Converts it to a DataFrame with the original raw feature names
    - Feeds it through the Pipeline
    """

    # Raw feature names used in the app / pipeline
    feature_names = FEATURE_NAMES

    # Background dataset: a single neutral row
    background_df = pd.DataFrame(
        [{name: 0 for name in feature_names}]
    )

    def model_fn(X):
        # X is a numpy array; convert to DataFrame with correct columns
        df = pd.DataFrame(X, columns=feature_names)
        # Pipeline handles preprocessing + prediction
        return model.predict_proba(df)[:, 1]

    explainer = shap.KernelExplainer(model_fn, background_df)
    return explainer
