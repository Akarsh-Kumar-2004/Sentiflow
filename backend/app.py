from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .model import SentimentModel
import pandas as pd

app = FastAPI(title="SentiFlow API")
model = SentimentModel()

class PredictionRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class FileBatchRequest(BaseModel):
    csv_path: str
    text_column: Optional[str] = "text"

@app.get("/")
def root():
    return {"message": "SentiFlow FastAPI is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    res = model.predict(req.text)
    return {
        "text": req.text,
        "label": res.get("label"),
        "score": float(res.get("score", 0.0)),
    }
#fixing the predict_batch endpoint to handle empty list and return proper error message
@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if len(req.texts) == 0:
        raise HTTPException(status_code=400, detail="texts list is empty")
    results = model.predict_batch(req.texts)
    output = []
    for text, res in zip(req.texts, results):
        output.append({
            "text": text,
            "label": res.get("label"),
            "score": float(res.get("score", 0.0)),
        })
    return {"items": output}

@app.post("/predict_csv")
def predict_csv(req: FileBatchRequest):
    try:
        df = pd.read_csv(req.csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load CSV: {e}")
    if req.text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.text_column}' not found")
    texts = df[req.text_column].astype(str).tolist()
    results = model.predict_batch(texts)
    df_out = df.copy()
    df_out["sentiment_label"] = [r.get("label") for r in results]
    df_out["sentiment_score"] = [float(r.get("score", 0.0)) for r in results]
    return {"predictions": df_out.to_dict(orient="records")}  
