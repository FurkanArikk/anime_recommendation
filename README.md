# Anime Recommender

End-to-end anime recommendation system over the MyAnimeList Top 10,000 dataset:
Gemini embeddings → Qdrant vector search → FastAPI → Streamlit chat UI.

> Work in progress. The full README is written in the final phase.

## Quick start (Docker only)

```bash
make env      # creates .env — fill in GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
make build
make up       # API on :8000 (docs at /docs), UI on :8501
```
