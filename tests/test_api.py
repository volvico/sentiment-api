import os, json, pathlib, sys
import time
from fastapi.testclient import TestClient
import pytest

# Allow import of app.api
BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from app.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("model_loaded") is True

def test_predict_basic():
    r = client.post("/predict", json={"texts": ["I love this product!", "This is bad."]})
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

# ---------- HEALTH ----------
def test_health_endpoint_status():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("model_loaded") is True
    assert data.get("mode") in ("pipeline", "separate")

# ---------- PREDICT (valid) ----------
def test_predict_endpoint_valid():
    """Cas normal mono-texte, réponse plate et proba cohérente."""
    test_data = {"text": "J'adore ce produit, il est fantastique !"}
    r = client.post("/predict_one", json=test_data)
    assert r.status_code == 200
    data = r.json()

    required = ["sentiment", "confidence", "probability_positive", "probability_negative"]
    for k in required:
        assert k in data

    assert data["sentiment"] in ["Positif", "Négatif"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0

    total = data["probability_positive"] + data["probability_negative"]
    assert abs(total - 1.0) < 0.01

# ---------- PREDICT (invalid) ----------
def test_predict_endpoint_invalid():
    # Vide
    r = client.post("/predict_one", json={"text": ""})
    assert r.status_code == 422

    # Trop long
    r = client.post("/predict_one", json={"text": "a" * 300})
    assert r.status_code == 422

    # JSON invalide
    r = client.post("/predict_one", json={})
    assert r.status_code == 422

# ---------- EXPLAIN (LIME) ----------
def test_explain_endpoint_lime():
    """Test LIME : HTML substantiel + explications."""
    test_data = {"text": "Ce film est absolument terrible, je le déteste !"}
    start = time.time()
    r = client.post("/explain_lime", json=test_data)
    duration = time.time() - start

    # Si LIME absent => 501 : le test documente aussi ce cas
    assert r.status_code in (200, 501)

    if r.status_code == 200:
        data = r.json()
        for k in ["sentiment", "confidence", "explanation", "html_explanation"]:
            assert k in data
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0
        assert isinstance(data["html_explanation"], str)
        assert len(data["html_explanation"]) > 100
        assert "<div" in data["html_explanation"]
        assert duration < 120

@pytest.mark.timeout(90)
def test_explain_robustness():
    """Robustesse LIME : textes courts/emoji/url"""
    for txt in ["Super !", "😊"*10, "http://example.com test"]:
        r = client.post("/explain_lime", json={"text": txt})
        # Autorise 501 si LIME non installé
        assert r.status_code in (200, 501)
        if r.status_code == 200:
            data = r.json()
            assert "html_explanation" in data

def test_predict_empty_list():
    r = client.post("/predict", json={"texts": []})
    assert r.status_code == 400

@pytest.mark.timeout(120)
def test_explain_lime_timeout_guard():
    r = client.post("/explain_lime", json={"text": "Un texte normal pour LIME"})
    assert r.status_code in (200, 501)
