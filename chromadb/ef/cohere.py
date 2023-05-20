from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class CohereEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "large"):
        try:
            import cohere
        except ImportError:
            raise ValueError(
                """The cohere python package is not installed.
                Please install it with `pip install cohere`
                """
            )

        self._client = cohere.Client(api_key)
        self._model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        # Call Cohere Embedding API for each document.
        return [
            embeddings
            for embeddings in self._client.embed(texts=texts, model=self._model_name)
        ]
