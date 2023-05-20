from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing import Any, Dict


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        if model_name not in self.models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ValueError(
                    """The sentence_transformers python package is not installed.
                    Please install it with `pip install sentence_transformers`
                    """
                )
            self.models[model_name] = SentenceTransformer(model_name, device=device)
        self._model = self.models[model_name]

    def __call__(self, texts: Documents) -> Embeddings:
        return self._model.encode(list(texts), convert_to_numpy=True).tolist()  # type: ignore # noqa E501
