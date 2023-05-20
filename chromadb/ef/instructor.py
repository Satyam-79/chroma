from chromadb.api.types import Documents, EmbeddingFunction, Embeddings


class InstructorEmbeddingFunction(EmbeddingFunction):
    # If you have a GPU with at least 6GB try
    # model_name = "hkunlp/instructor-xl" and device = "cuda"
    # for a list of models: https://github.com/HKUNLP/instructor-embedding#model-list
    def __init__(self, model_name: str = "hkunlp/instructor-base", device: str = "cpu"):
        try:
            from InstructorEmbedding import INSTRUCTOR
        except ImportError:
            raise ValueError(
                """The InstructorEmbedding python package is not installed.
                Please install it with `pip install InstructorEmbedding`
                """
            )
        self._model = INSTRUCTOR(model_name, device=device)

    def __call__(self, texts: Documents) -> Embeddings:
        return self._model.encode(texts).tolist()  # type: ignore
