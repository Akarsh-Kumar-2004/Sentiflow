# SentiFlow

SentiFlow is a real-time sentiment intelligence system with:
- FastAPI backend using HuggingFace DistilBERT sentiment model
- Streamlit frontend dashboard
- Batch CSV and manual text support
- Sentiment distribution, trends, word cloud

## Setup

1. Create Python venv
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. Install backend and frontend deps
```bash
pip install -r sentiflow/backend/requirements.txt
pip install -r sentiflow/frontend/requirements.txt
```

## Start backend

```bash
uvicorn sentiflow.backend.app:app --reload --port 8000
```

## Start frontend

```bash
streamlit run sentiflow/frontend/dashboard.py
```

## API endpoints
- `POST /predict` {"text": "some text"}
- `POST /predict_batch` {"texts": ["t1", "t2"]}
- `POST /predict_csv` {"csv_path": "path/to.csv", "text_column": "text"}
