from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class GooglePalmEmbeddingFunction(EmbeddingFunction):
    """To use this EmbeddingFunction, you must have the google.
    generativeai Python package installed and have a PaLM API key.
    """

    def __init__(self, api_key: str, model_name: str = "models/embedding-gecko-001"):
        if not api_key:
            raise ValueError("Please provide a PaLM API key.")

        if not model_name:
            raise ValueError("Please provide the model name.")

        try:
            import google.generativeai as palm
        except ImportError:
            raise ValueError(
                """The Google Generative AI python package is not installed.
                Please install it with `pip install google-generativeai`
                """
            )

        palm.configure(api_key=api_key)
        self._palm = palm
        self._model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        return [
            self._palm.generate_embeddings(model=self._model_name, text=text)[
                "embedding"
            ]
            for text in texts
        ]
