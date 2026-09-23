import openai
import cohere
from config import COHERE_MODELS, HF_MODEL_REGISTRY, OPENAI_MODELS


CUSTOM_MODEL_PREFIXES = {
    "huggingface": "HuggingFace Custom:",
    "openai": "OpenAI Custom:",
    "cohere": "Cohere Custom:",
}


def get_embedding_provider(model_name: str) -> str | None:
    if model_name in HF_MODEL_REGISTRY:
        return "huggingface"
    if model_name in OPENAI_MODELS:
        return "openai"
    if model_name in COHERE_MODELS:
        return "cohere"
    for provider, prefix in CUSTOM_MODEL_PREFIXES.items():
        if model_name.startswith(prefix) and model_name.removeprefix(prefix).strip():
            return provider
    return None


def _custom_model_id(model_name: str, provider: str) -> str | None:
    prefix = CUSTOM_MODEL_PREFIXES[provider]
    if not model_name.startswith(prefix):
        return None
    return model_name.removeprefix(prefix).strip()


class EmbeddingRunner:
    def __init__(self, openai_api_key: str = None, cohere_api_key: str = None, hf_token: str = None):
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        self.hf_token = hf_token
        self._st_models: dict = {}

    def _load_st_model(self, hf_model_id: str):
        if hf_model_id not in self._st_models:
            from sentence_transformers import SentenceTransformer
            self._st_models[hf_model_id] = SentenceTransformer(hf_model_id, token=self.hf_token)
        return self._st_models[hf_model_id]

    def embed(
        self,
        texts: list[str],
        model_name: str,
        input_type: str = "search_document",
    ) -> list[list[float]]:
        if model_name in HF_MODEL_REGISTRY:
            model = self._load_st_model(HF_MODEL_REGISTRY[model_name])
            return model.encode(texts, convert_to_numpy=True).tolist()

        hf_model_id = _custom_model_id(model_name, "huggingface")
        if hf_model_id:
            model = self._load_st_model(hf_model_id)
            return model.encode(texts, convert_to_numpy=True).tolist()

        if model_name in OPENAI_MODELS or model_name.startswith(CUSTOM_MODEL_PREFIXES["openai"]):
            openai_model_map = {
                "OpenAI text-embedding-3-small": "text-embedding-3-small",
                "OpenAI text-embedding-3-large": "text-embedding-3-large",
                "OpenAI text-embedding-ada-002": "text-embedding-ada-002",
            }
            api_model = openai_model_map.get(model_name) or _custom_model_id(model_name, "openai")
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(input=texts, model=api_model)
            return [item.embedding for item in response.data]

        if model_name in COHERE_MODELS or model_name.startswith(CUSTOM_MODEL_PREFIXES["cohere"]):
            cohere_model_map = {
                "Cohere Embed English v3.0": "embed-english-v3.0",
                "Cohere Embed Multilingual v3.0": "embed-multilingual-v3.0",
            }
            api_model = cohere_model_map.get(model_name) or _custom_model_id(model_name, "cohere")
            client = cohere.Client(api_key=self.cohere_api_key)
            response = client.embed(texts=texts, model=api_model, input_type=input_type)
            return [list(vec) for vec in response.embeddings]

        raise ValueError(f"Unknown model name: {model_name}")
