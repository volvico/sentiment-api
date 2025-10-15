# Déploiement complet : API FastAPI + UI Streamlit (Analyse de sentiments)

Cette partie 2 propose une **mise en production pédagogique** : une API REST (FastAPI) qui sert votre modèle, et une interface utilisateur (Streamlit) pour tester et expliquer les prédictions.

## Structure
```
sentiment_deploy/
├── app/
│   ├── api/
│   │   ├── main.py              # Serveur FastAPI (endpoints /health, /predict, /explain)
│   │   ├── schemas.py           # Schémas Pydantic (request/response)
│   │   ├── loader.py            # Chargement des artefacts (pipeline OU modèle + vectorizer)
│   │   └── explain.py           # Explicabilité légère (top tokens via coef_)
│   └── streamlit_app/
│       ├── app.py               # UI Streamlit
│       └── client.py            # Client HTTP vers l'API (requests)
├── api_artifacts/               # Placez ici les fichiers exportés à l'étape 1
│   ├── sentiment_pipeline.joblib       (OPTION 1 - recommandé)
│   ├── model_metadata.json
│   ├── emojis.json
│   ├── stopwords.json
│   ├── sentiment_model.joblib          (OPTION 2 - si pas de pipeline)
│   └── tfidf_vectorizer.joblib         (OPTION 2)
├── tests/
│   └── test_api.py              # Tests pytest sur l'API
├── requirements.txt
├── .env.sample
└── Makefile
```

## Pré-requis
- Python 3.10+ (recommandé)
- `api_artifacts/` doit contenir soit :
  - **Option 1 (recommandée)** : `sentiment_pipeline.joblib` + `model_metadata.json`
  - **Option 2** : `sentiment_model.joblib` + `tfidf_vectorizer.joblib` + `model_metadata.json`

## Installation & Lancement (local)
```bash
cd sentiment_deploy
python -m venv .venv && source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
cp .env.sample .env

# Lancer l'API
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Lancer l'UI (dans un autre terminal)
streamlit run app/streamlit_app/app.py --server.port 8501
```

## Tests
```bash
pytest -q
```

## Explications rapides
- L'API charge d'abord le **pipeline** s'il est présent. Sinon, elle charge **modèle + vectorizer** et applique la fonction `preprocess_text` référencée dans `model_metadata.json` si fournie (module + nom).  
- L'endpoint `/explain` renvoie les **tokens les plus influents** pour la prédiction (approx. via `coef_` d'une régression logistique).

Bon TP !
