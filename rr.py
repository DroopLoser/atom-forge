from fastapi import FastAPI
from schemas.input import MoleculeInput
from fastapi.middleware.cors import CORSMiddleware
from rdkit import Chem
from rdkit.Chem import QED, Crippen
import sascorer   # important

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(data: MoleculeInput):

    smiles = data.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {"error": "Invalid SMILES", "smiles": smiles}

    return {
        "smiles": smiles,
        "logp": float(Crippen.MolLogP(mol)),
        "sas": float(sascorer.calculateScore(mol)),
        "qed": float(QED.qed(mol))
    }
