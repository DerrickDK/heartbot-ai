from typing import Dict, Any

import gradio as gr

from src.config import FEATURE_NAMES
from src.heartbot import HeartBot

def build_interface() -> gr.Blocks:
    """
    Build for the Gradio UI.
    """

    bot = HeartBot()
    
    error_css = """
    .invalid, :invalid, .error, .textfield.invalid, input:invalid {
    border: 2px solid #ef4444 !important;
    background-color: transparent !important;
    }
    """

    with gr.Blocks(title="HeartBot-AI") as demo:
        gr.Markdown("# **HeartBot-AI**: A Heart Disease Prediction and Patient Support Chatbot")

        with gr.Tab("Risk Prediction"):
            gr.Markdown("**Instructions**: Enter your clinical details to estimate the risk of heart disease.")
            gr.Markdown("**Note**: Heartbot-ai uses a trained Logistic Regression model and SHAP explainability.")

            inputs = {}
            with gr.Row():
                inputs["age"] = gr.Number(
                    label="Age",
                    info="Age must be between 1 and 120",
                    minimum=1,
                    maximum=120
                )

                inputs["sex"] = gr.Dropdown(
                    label="Sex",
                    choices=[0, 1],
                    value=1,
                    info="0 = Female, 1 = Male"
                )

                inputs["cp"] = gr.Dropdown(
                    label="Chest Pain Type (cp)",
                    choices=[0, 1, 2, 3],
                    value=0,
                    info="0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic"
                )

            with gr.Row():
                inputs["trestbps"] = gr.Number(
                    label="Resting Blood Pressure (trestbps)",
                    info="Resting blood pressure must be between 50 and 250",
                    minimum=50,
                    maximum=250
                )

                inputs["chol"] = gr.Number(
                    label="Cholesterol (chol)",
                    info="Serum cholesterol must be between 100 and 600",
                    minimum=100,
                    maximum=600
                )

                inputs["fbs"] = gr.Dropdown(
                    label="Fasting Blood Sugar > 120 mg/dl (fbs)",
                    choices=[0, 1],
                    value=0,
                    info="0 = False, 1 = True"
                )

            with gr.Row():
                inputs["restecg"] = gr.Dropdown(
                    label="Resting ECG (restecg)",
                    choices=[0, 1, 2],
                    value=0,
                    info="0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy"
                )

                inputs["thalach"] = gr.Number(
                    label="Max Heart Rate Achieved (thalach)",
                    info="Maximum heart rate achieved between 50 and 250",
                    minimum=50,
                    maximum=250
                )

                inputs["exang"] = gr.Dropdown(
                    label="Exercise-Induced Angina (exang)",
                    choices=[0, 1],
                    value=0,
                    info="0 = No, 1 = Yes"
                )

            with gr.Row():
                inputs["oldpeak"] = gr.Number(
                    label="ST Depression (oldpeak)",
                    info="ST depression induced by exercise relative to rest between 0 and 10",
                    minimum=0.0,
                    maximum=10.0
                )

                inputs["slope"] = gr.Dropdown(
                    label="Slope of ST Segment (slope)",
                    choices=[0, 1, 2],
                    value=1,
                    info="0 = Upsloping, 1 = Flat, 2 = Downsloping"
                )

            with gr.Row():
                inputs["ca"] = gr.Dropdown(
                    label="Number of Major Vessels (ca)",
                    choices=[0, 1, 2, 3, 4],
                    value=0,
                    info="Number of major vessels colored by fluoroscopy"
                )

                inputs["thal"] = gr.Dropdown(
                    label="Thalassemia (thal)",
                    choices=[2, 3, 7],
                    value=2,
                    info="2 = Normal, 3 = Fixed defect, 7 = Reversible defect"
                )


            predict_button = gr.Button("Predict Risk")
            clear_btn = gr.Button("Clear")
            pred_output_label = gr.Textbox(label="Prediction", interactive=False)
            pred_output_proba = gr.Textbox(label="Probability", interactive=False)
            shap_output = gr.DataFrame(label="SHAP Feature Level Explanation", interactive=False)
            clear_btn.click(
                fn=lambda: ("", "", ""),
                inputs=[], 
                outputs=[pred_output_label, pred_output_proba, shap_output]
                )

            def on_predict(*vals):
                features: Dict[str, Any] = {
                    feature: vals[i]
                    for i, feature in enumerate(FEATURE_NAMES)
                }
                result = bot.handle_prediction(features)
                if result["type"] == "error":
                    return result["message"], "", {}
                return (
                    result["label"],
                    f"{result['probability']* 100:.2f}%",
                    result["shap_table"],
                )

            predict_button.click(
                fn=on_predict,
                inputs=[inputs[f] for f in FEATURE_NAMES],
                outputs=[pred_output_label, pred_output_proba, shap_output],
            )

        with gr.Tab("Ask a Question"):
            gr.Markdown(
                "Ask any question about heart disease, your clinical information, or specific model features."
            )

            def on_rag(query: str, history):
                result = bot.handle_rag_query(query)

                answer = result["answer"]
                sources = result.get("sources", [])

                if sources:
                    formatted_sources = "\n\n**Sources:**\n" + "\n".join([f"- {src}" for src in sources])
                else:
                    formatted_sources = "\n\n**Sources:** None found."

                return answer + formatted_sources

            
            gr.ChatInterface(
                fn=on_rag,
                examples=[
                    "What is heart disease?",
                    "How do I know if I have heart disease?",
                    "What are the symptoms of heart disease?"
                ],
                textbox=gr.Textbox(placeholder="Ask a question about heart disease or the information you provided...", container=False, scale=7),
                chatbot=gr.Chatbot(placeholder="Hello! I'm HeartBot-AI. How can I help you today?"),
                cache_examples=False,
                flagging_mode="never",
                autofocus=True,
                save_history=False
            )

    return demo

