from functools import lru_cache

import nltk
import numpy as np
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


@lru_cache(maxsize=1)
def _get_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _semantic_chunks(text: str, chunk_size: int, chunk_overlap: int, breakpoint_percentile: int) -> list[str]:
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    sentences = nltk.tokenize.sent_tokenize(text)
    if len(sentences) < 2:
        return sentences or ([text] if text.strip() else [])

    embeddings = _get_semantic_model().encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    distances = 1 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks = []
    current = [sentences[0]]
    for index, sentence in enumerate(sentences[1:], start=1):
        candidate_size = len(" ".join(current)) + len(sentence) + 1
        semantic_break = distances[index - 1] > threshold
        if semantic_break or candidate_size > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    chunks.append(" ".join(current))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [part for chunk in chunks for part in splitter.split_text(chunk)]


def chunk_text(text: str, strategy: str, chunk_size: int = 512, chunk_overlap: int = 50, **kwargs) -> list[str]:
    if strategy == "Fixed-Size (512)":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="",
        )
        return splitter.split_text(text)

    if strategy == "Recursive Character":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(text)

    if strategy == "Markdown Header-Aware":
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_docs = md_splitter.split_text(text)
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = []
        for doc in md_docs:
            content = doc.page_content if hasattr(doc, "page_content") else doc
            if len(content) > chunk_size:
                chunks.extend(secondary.split_text(content))
            else:
                chunks.append(content)
        return chunks

    if strategy == "Semantic Splitting":
        return _semantic_chunks(
            text,
            chunk_size,
            chunk_overlap,
            kwargs.get("breakpoint_percentile", 80),
        )

    if strategy == "Sentence Window":
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        sentences = nltk.tokenize.sent_tokenize(text)
        window_size = kwargs.get("window_size", 5)
        overlap_sentences = kwargs.get("overlap_sentences", 1)
        chunks = []
        i = 0
        while i < len(sentences):
            if i == 0:
                group = sentences[i:i + window_size]
            else:
                start = max(0, i - overlap_sentences)
                group = sentences[start:i + window_size]
            chunks.append(" ".join(group))
            i += window_size
        return chunks

    if strategy == "Token-based (tiktoken)":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            window = tokens[start:end]
            chunks.append(enc.decode(window))
            if end >= len(tokens):
                break
            start += chunk_size - chunk_overlap
        return chunks

    if strategy == "Sliding Window":
        overlap_pct = kwargs.get("overlap_pct", 50)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * overlap_pct / 100),
        )
        return splitter.split_text(text)

    if strategy == "Paragraph-based":
        min_paragraph_len = kwargs.get("min_paragraph_len", 20)
        paragraphs = text.split("\n\n")
        paragraphs = [p for p in paragraphs if len(p) >= min_paragraph_len]
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = []
        for para in paragraphs:
            if len(para) > chunk_size * 2:
                chunks.extend(secondary.split_text(para))
            else:
                chunks.append(para)
        return chunks

    if strategy == "Table-Aware":
        lines = text.splitlines()
        blocks = []
        i = 0
        while i < len(lines):
            block_lines = []
            while i < len(lines):
                line = lines[i]
                is_table_line = line.count("|") >= 2 or line.count(",") >= 3
                block_lines.append((line, is_table_line))
                i += 1
                if block_lines and (block_lines[-1][1] != block_lines[0][1]):
                    i -= 1
                    block_lines.pop()
                    break
            if block_lines:
                is_table_block = block_lines[0][1]
                raw_lines = [bl[0] for bl in block_lines]
                table_line_count = sum(1 for bl in block_lines if bl[1])
                if len(block_lines) > 0 and table_line_count / len(block_lines) >= 0.6:
                    is_table_block = True
                else:
                    is_table_block = False
                blocks.append((is_table_block, "\n".join(raw_lines)))

        merged = []
        for is_table, content in blocks:
            if merged and merged[-1][0] == is_table:
                merged[-1] = (is_table, merged[-1][1] + "\n" + content)
            else:
                merged.append([is_table, content])

        chunks = []
        secondary = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for is_table, content in merged:
            if not is_table:
                chunks.extend(secondary.split_text(content))
            else:
                if len(content) <= chunk_size * 3:
                    chunks.append(content)
                else:
                    table_lines = content.splitlines()
                    header = table_lines[0] if table_lines else ""
                    data_lines = table_lines[1:] if len(table_lines) > 1 else []
                    rows_per_group = max(1, chunk_size // 80)
                    for j in range(0, len(data_lines), rows_per_group):
                        group = data_lines[j:j + rows_per_group]
                        chunk_content = header + "\n" + "\n".join(group)
                        chunks.append(chunk_content)
        return [c for c in chunks if c.strip()]

    if strategy == "Code-Aware":
        separators = [
            "\nclass ",
            "\ndef ",
            "\n\tdef ",
            "\n  def ",
            "\n    def ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        return splitter.split_text(text)

    if strategy == "Parent-Child (Hierarchical)":
        parent_size = kwargs.get("parent_chunk_size", 1024)
        child_size = kwargs.get("child_chunk_size", 256)
        child_overlap_val = kwargs.get("child_overlap", 30)

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=0,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap_val,
        )

        parent_chunks = parent_splitter.split_text(text)
        all_chunks = []
        for parent in parent_chunks:
            children = child_splitter.split_text(parent)
            for child in children:
                all_chunks.append(f"[PARENT] {parent[:100]}...\n[CHILD] {child}")
        return all_chunks

    raise ValueError(f"Unknown chunking strategy: {strategy}")
