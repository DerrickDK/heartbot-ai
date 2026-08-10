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
        
    def _build_shap_table(self, shap_values, df_input):
        """
        Convert raw SHAP values into a readable feature-level explanation table.
        """

        feature_names = df_input.columns

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0],
            "Impact": [
                "Increases risk" if v > 0 else
                "Decreases risk" if v < 0 else
                "No impact"
                for v in shap_values[0]
            ]
        })

        # Sort by absolute impact
        shap_df["AbsImpact"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values("AbsImpact", ascending=False)

        # Remove helper column
        shap_df = shap_df.drop(columns=["AbsImpact"])

        return shap_df

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
            
            shap_table = self._build_shap_table(shap_values, df_input)

            return {
                "prediction": int(pred),
                "probability": float(proba),
                "shap_table": shap_table
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
