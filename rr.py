from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running"}

@app.post("/predict")
def predict(data: dict):
    return {
        "smiles": data["smiles"],
        "logp": 1.2,
        "sas": 0.5,
        "qed": 0.8
    }