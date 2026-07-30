import pandas as pd
from src.model_loader import load_model, init_shap_explainer
from src.config import HEART_MODEL_PATH

class PredictionService:
    """
    Handles prediction + SHAP explainability using the Pipeline model.
    """

    def __init__(self):
        self.model = load_model(HEART_MODEL_PATH)
        self.explainer = init_shap_explainer(self.model)

    def predict(self, features):
        """
        Predict heart disease risk using the Pipeline model.
        """

        try:
            df_input = pd.DataFrame([features])

            # Prediction via Pipeline
            pred = self.model.predict(df_input)[0]
            proba = self.model.predict_proba(df_input)[0][1]

            # SHAP values: pass raw input as numpy to KernelExplainer
            shap_values = self.explainer.shap_values(df_input.values)

            return {
                "prediction": int(pred),
                "probability": float(proba),
                "shap_values": shap_values.tolist()
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
