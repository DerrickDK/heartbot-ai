from typing import Dict, Any

from src.predictor import PredictionService
from src.rag_pipeline import RAGPipeline


class HeartBot:
    """
    High-level chatbot logic that routes between:
    - Structured prediction (heart disease risk)
    - RAG-based Q&A (medical / dataset questions)
    """

    def __init__(self):
        self.prediction_service = PredictionService()
        self.rag_pipeline = RAGPipeline()

    def handle_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle structured prediction request.
        """
        result = self.prediction_service.predict(features)
        if "error" in result:
            return {"type": "error", "message": result["error"]}

        pred_label = "Heart disease likely" if result["prediction"] == 1 else "Heart disease unlikely"

        return {
            "type": "prediction",
            "label": pred_label,
            "probability": result["probability"],
            "shap_values": result["shap_values"],
        }

    def handle_rag_query(self, query: str) -> Dict[str, Any]:
        """
        Handle natural language RAG query.
        """
        rag_result = self.rag_pipeline.query(query)
        return {
            "type": "rag",
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
        }
