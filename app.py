from src.ui import build_interface


def main():
    """
    Main entry point to launch the Gradio app.
    """
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()

