"""
Lightweight local retrieval over the notes in ./knowledge (TF-IDF, scikit-learn).
No downloads, no GPU, no embeddings server. The notes are project notes written
for CONTEXTRA - verify claims against primary sources before citing them.
"""

from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _load_chunks(directory=None):
    chunks = []
    for path in sorted(Path(directory or KNOWLEDGE_DIR).glob("*.md")):
        title = path.stem
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                title = block.lstrip("# ").strip()
                continue
            chunks.append({"source": path.name, "title": title, "text": block})
    return chunks


class KnowledgeBase:
    def __init__(self, directory=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = _load_chunks(directory)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def retrieve(self, query, k=3, min_score=0.05):
        if not self.chunks or not query.strip():
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [
            {**self.chunks[i], "score": round(float(scores[i]), 3)}
            for i in order if scores[i] >= min_score
        ]


@lru_cache(maxsize=1)
def get_kb():
    return KnowledgeBase()
