import re
import atexit
import shutil
import tempfile
import numpy as np


def _safe_name(config_key: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9-]", "-", config_key)[:60]
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) < 3:
        name = f"col-{name}"
    return name[:63]


class VectorStoreManager:
    def __init__(self):
        import chromadb
        self._tmp_dir = tempfile.mkdtemp(prefix="embedeval_chroma_")
        atexit.register(shutil.rmtree, self._tmp_dir, ignore_errors=True)
        self._client = chromadb.PersistentClient(path=self._tmp_dir)

    @property
    def storage_path(self) -> str:
        return self._tmp_dir

    def upsert(self, config_key: str, chunks: list[str], embeddings: np.ndarray):
        name = _safe_name(config_key)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        col = self._client.create_collection(
            name,
            metadata={"hnsw:space": "cosine", "config_key": config_key},
        )
        col.add(
            ids=[f"c{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist(),
        )

    def delete(self, config_key: str):
        try:
            self._client.delete_collection(_safe_name(config_key))
        except Exception:
            pass

    def delete_all(self):
        for col in self._client.list_collections():
            try:
                self._client.delete_collection(col.name)
            except Exception:
                pass

    def list_configs(self) -> list[dict]:
        results = []
        for col in self._client.list_collections():
            try:
                c = self._client.get_collection(col.name)
                results.append({
                    "config": col.metadata.get("config_key", col.name),
                    "collection_name": col.name,
                    "count": c.count(),
                })
            except Exception:
                pass
        return results

    def total_chunks(self) -> int:
        return sum(c["count"] for c in self.list_configs())
