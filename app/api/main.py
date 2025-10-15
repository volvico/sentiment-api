import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    HealthResponse, PredictRequest, PredictResponse, PredictItem,
    ExplainRequest, ExplainResponse, TokenContribution,
    PredictOneRequest, PredictOneResponse,
    ExplainLimeRequest, ExplainLimeResponse
)
from .loader import Artifacts
from .explain import top_tokens_by_coef

app = FastAPI(title="Sentiment API", version="1.0.0", description="REST API pour prédire des sentiments")

# CORS pour autoriser Streamlit local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ART = Artifacts()
ART.load()

@app.get("/health", response_model=HealthResponse)
def health():
    model_class = None
    vectorizer_class = None
    if ART.mode == "pipeline" and ART.pipeline is not None:
        try:
            model = ART.pipeline.named_steps.get("model")
            vect = ART.pipeline.named_steps.get("tfidf") or ART.pipeline.named_steps.get("vectorizer")
            model_class = model.__class__.__name__ if model else None
            vectorizer_class = vect.__class__.__name__ if vect else None
        except Exception:
            pass
    elif ART.mode == "separate":
        model_class = ART.model.__class__.__name__ if ART.model else None
        vectorizer_class = ART.vectorizer.__class__.__name__ if ART.vectorizer else None
    return HealthResponse(status="ok", model_loaded=True, mode=ART.mode, model_class=model_class, vectorizer_class=vectorizer_class)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")
    y, proba = ART.predict(req.texts)
    items = []
    for i, t in enumerate(req.texts):
        label = str(y[i])
        probs = None
        if proba is not None:
            probs = [float(x) for x in proba[i]]
        items.append(PredictItem(text=t, label=label, proba=probs))
    return PredictResponse(predictions=items)

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    # token contributions for the given text (using vectorizer)
    toks = ART.tokens_for(req.text)
    # top +/- tokens via model coef
    if ART.mode == "pipeline":
        model = ART.pipeline.named_steps.get("model")
        vect  = ART.pipeline.named_steps.get("tfidf") or ART.pipeline.named_steps.get("vectorizer")
    else:
        model = ART.model
        vect  = ART.vectorizer
    tops = top_tokens_by_coef(model, vect, k=10)
    return ExplainResponse(
        tokens=[TokenContribution(token=t, weight=w) for t, w in toks],
        top_positive=[TokenContribution(token=t, weight=w) for t, w in tops.get("pos", [])],
        top_negative=[TokenContribution(token=t, weight=w) for t, w in tops.get("neg", [])],
    )

# ---------- /predict_one (mono-texte) ----------
@app.post("/predict_one", response_model=PredictOneResponse)
def predict_one(req: PredictOneRequest):
    # Recyclage du moteur existant
    y, proba = ART.predict([req.text])
    label = str(y[0])

    # Probabilités binaires : si modèle ne fournit rien, approx simple
    if proba is not None:
        probs = proba[0]
        if len(probs) == 2:
            p_pos = float(probs[1])
            p_neg = float(probs[0])
        else:
            # multiclasse: confidence = prob de la classe prédite, pas de split pos/neg
            idx = int(y[0]) if isinstance(y[0], (int, float)) else 0
            p_pos = float(probs[idx])
            p_neg = 1.0 - p_pos
    else:
        # fallback neutre si pas de predict_proba
        p_pos = 0.5
        p_neg = 0.5

    confidence = max(p_pos, p_neg)

    # Mapper label → "Positif"/"Négatif" si besoin
    # Essai heuristique: si classes_ disponible et binaire {0,1}
    readable = label
    try:
        classes = None
        if ART.mode == "pipeline":
            model = ART.pipeline.named_steps.get("model")
            classes = getattr(model, "classes_", None)
        else:
            classes = getattr(ART.model, "classes_", None)
        if classes is not None and len(classes) == 2:
            # suppose 1 = Positif
            if str(y[0]) == str(classes[1]):
                readable = "Positif"
            else:
                readable = "Négatif"
        else:
            # Si pas d'info, heuristique probas
            readable = "Positif" if p_pos >= p_neg else "Négatif"
    except Exception:
        readable = "Positif" if p_pos >= p_neg else "Négatif"

    return PredictOneResponse(
        sentiment=readable,
        confidence=float(confidence),
        probability_positive=float(p_pos),
        probability_negative=float(p_neg),
    )

# ---------- /explain_lime (mono-texte, avec HTML) ----------
@app.post("/explain_lime", response_model=ExplainLimeResponse)
def explain_lime(req: ExplainLimeRequest):
    # LIME est optionnel (install lime)
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception:
        # On peut renvoyer 501 si LIME pas installé
        raise HTTPException(status_code=501, detail="LIME non installé. Installez le package 'lime'.")

    text = req.text

    # 🔒 Garde: si pas de tokens "mots", LIME plante (ex: uniquement emojis)
    if not re.search(r"\w", text):
        p1 = predict_one(PredictOneRequest(text=text))
        return ExplainLimeResponse(
            sentiment=p1.sentiment,
            confidence=p1.confidence,
            explanation=[],
            html_explanation="<div>Aucun token exploitable pour LIME.</div>"
        )

    # Fonction prédictive adaptée au format attendu par LIME: proba par classe
    def predict_proba_fn(texts):
        _, proba = ART.predict(texts)
        if proba is None:
            # si le modèle ne supporte pas predict_proba, fabriquer une proba neutre
            import numpy as np
            return np.tile([0.5, 0.5], (len(texts), 1))
        return proba

    # Noms de classes (si disponibles)
    class_names = None
    try:
        if ART.mode == "pipeline":
            model = ART.pipeline.named_steps.get("model")
            classes = getattr(model, "classes_", None)
        else:
            classes = getattr(ART.model, "classes_", None)
        if classes is not None and len(classes) == 2:
            class_names = [str(classes[0]), str(classes[1])]
    except Exception:
        pass

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba_fn, num_features=10)

    # Déterminer sentiment/confiance depuis predict_one pour cohérence
    p1 = predict_one(PredictOneRequest(text=text))

    # HTML
    html = exp.as_html()

    # Liste lisible (feature, weight)
    readable_expl = [f"{feat}: {weight:.3f}" for feat, weight in exp.as_list()]

    return ExplainLimeResponse(
        sentiment=p1.sentiment,
        confidence=p1.confidence,
        explanation=readable_expl,
        html_explanation=html
    )
