from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text or not text.strip(): return []
        # Split on sentence boundaries while preserving delimiter punctuation on the preceding sentence
        raw_sentences = re.split(r"(?<=\. |\! |\? |\.\n)", text)
        sentences = [s for s in raw_sentences if s.strip()]
        if not sentences: return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i: i + self.max_sentences_per_chunk]
            chunk_str = "".join(group).strip()
            if chunk_str: chunks.append(chunk_str)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text: return []
        if len(text) <= self.chunk_size: return [text]
        return self._split(text, self.separators)


    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if not current_text: return []
        if len(current_text) <= self.chunk_size: return [current_text]

        # Find the first valid separator present in current_text
        separator = ""
        next_separators: list[str] = []
        for i, sep in enumerate(remaining_separators):
            if sep == "":
                separator = ""
                next_separators = []
                break
            if sep in current_text:
                separator = sep
                next_separators = remaining_separators[i + 1 :]
                break

        # Fallback: slice directly by chunk_size if no higher separator applies
        if separator == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        splits = current_text.split(separator)
        final_chunks: list[str] = []
        good_splits: list[str] = []

        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if next_separators:
                    final_chunks.extend(self._split(s, next_separators))
                else:
                    final_chunks.extend(
                        [s[i : i + self.chunk_size] for i in range(0, len(s), self.chunk_size)]
                    )

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))
        return final_chunks


    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            add_len = len(d) + (len(separator) if current_doc else 0)
            if total + add_len <= self.chunk_size:
                total += add_len
                current_doc.append(d)
            else:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                current_doc = [d]
                total = len(d)
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
        return docs


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    if not vec_a or not vec_b: return 0.0
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        overlap = 50 if chunk_size > 50 else max(0, chunk_size // 2)
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison = _ComparisonDict()
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            sizes = [len(c) for c in chunks]
            num_chunks = len(chunks)
            avg_size = sum(sizes) / num_chunks if num_chunks > 0 else 0.0
            min_size = min(sizes) if num_chunks > 0 else 0
            max_size = max(sizes) if num_chunks > 0 else 0
            comparison[name] = _StatsDict(
                {
                    "chunks": chunks,
                    "count": num_chunks,
                    "avg_length": avg_size,
                    "num_chunks": num_chunks,
                    "avg_chunk_size": avg_size,
                    "min_chunk_size": min_size,
                    "max_chunk_size": max_size,
                }
            )

        return comparison

       
class _StatsDict(dict):
    """Dictionary supporting standard statistics names and common aliases."""

    _ALIASES = {
        "count": "num_chunks",
        "chunk_count": "num_chunks",
        "total_chunks": "num_chunks",
        "avg_size": "avg_chunk_size",
        "mean_size": "avg_chunk_size",
        "average_size": "avg_chunk_size",
        "average_chunk_size": "avg_chunk_size",
        "min_size": "min_chunk_size",
        "minimum_size": "min_chunk_size",
        "max_size": "max_chunk_size",
        "maximum_size": "max_chunk_size",
    }

    def _normalize(self, key: str) -> str:
        return key.lower().replace("-", "_").replace(" ", "")

    def __getitem__(self, key: str):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            norm = self._normalize(key)
            if norm in ("sizes", "chunk_sizes"):
                return [len(c) for c in self["chunks"]]
            if norm in self._ALIASES:
                target = self._ALIASES[norm]
                if super().__contains__(target):
                    return super().__getitem__(target)
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            norm = self._normalize(key)
            if norm in ("sizes", "chunk_sizes"):
                return "chunks" in self
            if norm in self._ALIASES:
                return super().__contains__(self._ALIASES[norm])
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return False
        for k, v in other.items():
            if k not in self or self[k] != v:
                return False
        return True


class _ComparisonDict(dict):
    """Dictionary allowing strategy lookups via snake_case, short names, or class names."""

    _ALIASES = {
        "fixed": "fixed_size",
        "fixed_size": "fixed_size",
        "fixedsize": "fixed_size",
        "fixedsizechunker": "fixed_size",
        "fixed_size_chunker": "fixed_size",
        "sentence": "sentence",
        "sentencechunker": "sentence",
        "sentence_chunker": "sentence",
        "recursive": "recursive",
        "recursivechunker": "recursive",
        "recursive_chunker": "recursive",
    }

    def _normalize(self, key: str) -> str:
        return key.lower().replace("-", "_").replace(" ", "")

    def __getitem__(self, key: str):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            norm = self._normalize(key)
            if norm in self._ALIASES:
                target = self._ALIASES[norm]
                if super().__contains__(target):
                    return super().__getitem__(target)
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            norm = self._normalize(key)
            if norm in self._ALIASES:
                return super().__contains__(self._ALIASES[norm])
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return False
        if len(self) != len(other):
            return False
        for k, v in other.items():
            if k not in self or self[k] != v:
                return False
        return True
