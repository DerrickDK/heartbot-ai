from src.ui import build_interface, gr, ERROR_CSS


if __name__ == "__main__":
    """
    Main entry point to launch the Gradio app.
    """
    demo = build_interface()
    demo.launch(theme=gr.themes.Soft(), css=ERROR_CSS)

