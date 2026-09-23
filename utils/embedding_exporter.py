import io
import json
import tempfile
import os
import numpy as np


def export_embeddings_npz(embedding_store: dict) -> bytes:
    buf = io.BytesIO()
    arrays = {}
    for config_key, data in embedding_store.items():
        safe = config_key.replace(" | ", "__").replace(" ", "_").replace("/", "-")[:60]
        arrays[f"{safe}__embeddings"] = data["embeddings"]
        arrays[f"{safe}__chunks"] = np.array(data["chunks"], dtype=object)
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def export_faiss_index(embedding_store: dict, config_key: str) -> bytes:
    import faiss
    data = embedding_store[config_key]
    emb = np.array(data["embeddings"], dtype=np.float32, copy=True)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    with open(tmp_path, "rb") as f:
        result = f.read()
    os.unlink(tmp_path)
    return result


def export_embeddings_jsonl(embedding_store: dict, config_key: str) -> bytes:
    data = embedding_store[config_key]
    lines = [
        json.dumps({"text": chunk, "embedding": vec, "config": config_key})
        for chunk, vec in zip(data["chunks"], data["embeddings"].tolist())
    ]
    return "\n".join(lines).encode("utf-8")


def get_store_summary(embedding_store: dict) -> dict:
    total_vectors = sum(d["num_chunks"] for d in embedding_store.values())
    configs = [
        {
            "config": k,
            "model": v["model"],
            "strategy": v["strategy"],
            "num_chunks": v["num_chunks"],
            "dimensions": v["dimensions"],
            "size_mb": round((v["num_chunks"] * v["dimensions"] * 4) / (1024 * 1024), 2),
        }
        for k, v in embedding_store.items()
    ]
    return {"total_vectors": total_vectors, "configs": configs}
