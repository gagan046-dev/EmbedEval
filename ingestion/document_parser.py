import io
import pandas as pd


def parse_document(file_bytes: bytes, file_name: str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        import pymupdf as fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)

    if ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")

    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes)).to_string(index=False)

    if ext == "xlsx":
        return pd.read_excel(io.BytesIO(file_bytes)).to_string(index=False)

    raise ValueError(f"Unsupported file type: .{ext}")
