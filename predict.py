from fastapi import FastAPI
import joblib

app = FastAPI()

dv, model = joblib.load("model.joblib")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: dict):
    X = dv.transform([customer])
    prob = model.predict_proba(X)[0,1]
    return {"default_probability": float(prob)}