import numpy as np


def build_faiss_index(embeddings: list[list[float]]):
    import faiss
    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_top_k(
    query: str,
    chunks: list[str],
    index,
    embedding_runner,
    model_name: str,
    k: int = 5,
) -> list[str]:
    query_vector = np.array(
        embedding_runner.embed([query], model_name, input_type="search_query"),
        dtype=np.float32,
    )
    import faiss
    faiss.normalize_L2(query_vector)
    _, indices = index.search(query_vector, min(k, len(chunks)))
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
