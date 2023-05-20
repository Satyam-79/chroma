from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        try:
            import requests
        except ImportError:
            raise ValueError(
                """The requests python package is not installed.
                    Please install it with `pip install requests`
                """
            )
        self._api_url = f"""
            https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}
            """
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __call__(self, texts: Documents) -> Embeddings:
        # Call HuggingFace Embedding API for each document
        return self._session.post(  # type: ignore
            self._api_url, json={"inputs": texts, "options": {"wait_for_model": True}}
        ).json()
