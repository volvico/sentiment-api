import os, json, importlib, joblib, numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "./api_artifacts"))

class Artifacts:
    def __init__(self):
        self.mode = "none"  # 'pipeline' or 'separate'
        self.pipeline = None
        self.model = None
        self.vectorizer = None
        self.preprocess_fn: Optional[Callable[[str], str]] = None
        self.meta = {}

    @staticmethod
    def _try_load(path: Path):
        return joblib.load(path) if path.exists() else None

    def load(self):
        meta_path = ARTIFACTS_DIR / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        # 1) Try pipeline
        pipe_path = ARTIFACTS_DIR / "sentiment_pipeline.joblib"
        self.pipeline = self._try_load(pipe_path)
        if self.pipeline is not None:
            self.mode = "pipeline"
            return

        # 2) Fallback: separate pieces
        self.model = self._try_load(ARTIFACTS_DIR / "sentiment_model.joblib")
        self.vectorizer = self._try_load(ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
        if self.model is not None and self.vectorizer is not None:
            self.mode = "separate"
        else:
            raise FileNotFoundError("Aucun artefact modèle/pipeline trouvé dans api_artifacts/")

        # Recreate preprocess function from meta if provided
        fn_name = self.meta.get("preprocess_fn_name")
        fn_module = self.meta.get("preprocess_fn_module")
        if fn_name and fn_module:
            try:
                mod = importlib.import_module(fn_module)
                self.preprocess_fn = getattr(mod, fn_name, None)
            except Exception:
                self.preprocess_fn = None

    def predict(self, texts):
        if self.mode == "pipeline":
            pipe = self.pipeline
            y = pipe.predict(texts)
            proba = pipe.predict_proba(texts) if hasattr(pipe, "predict_proba") else None
            return y, proba
        else:
            # separate
            if self.preprocess_fn:
                texts = [self.preprocess_fn(t) for t in texts]
            X = self.vectorizer.transform(texts)
            y = self.model.predict(X)
            proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
            return y, proba

    def tokens_for(self, text: str):
        """Return vectorized tokens for a single text (used for explanations)."""
        if self.mode == "pipeline":
            # try to access vectorizer inside pipeline
            try:
                vect = self.pipeline.named_steps.get("tfidf") or self.pipeline.named_steps.get("vectorizer")
            except Exception:
                vect = None
        else:
            vect = self.vectorizer
        if vect is None:
            return []
        if self.preprocess_fn and self.mode == "separate":
            text = self.preprocess_fn(text)
        X = vect.transform([text])
        # map indices back to terms (works for sklearn TF-IDF)
        if hasattr(vect, "get_feature_names_out"):
            feats = vect.get_feature_names_out()
        else:
            # older sklearn
            feats = np.array(sorted(vect.vocabulary_, key=vect.vocabulary_.get))
        row = X.tocoo()
        return [(feats[j], float(v)) for i,j,v in zip(row.row, row.col, row.data) if i == 0]
