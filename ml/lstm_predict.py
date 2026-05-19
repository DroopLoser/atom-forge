import tensorflow as tf
import os
from rdkit import Chem
from ml.preprocess import encode_smiles

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully ✔️")


def predict_smiles(smiles: str):
    print("SMILES:", smiles)

    mol = Chem.MolFromSmiles(smiles)
    print("MOL:", mol)

    input_data = encode_smiles(smiles)
    print("INPUT SHAPE:", getattr(input_data, "shape", None))
    print("INPUT DATA:", input_data)

    prediction = model.predict(input_data)[0]
    print("RAW PREDICTION:", prediction)

    return {
    "logp": float(prediction[0]),
    "sas": float(prediction[1]),
    "qed": max(0, min(1, float(prediction[2])))
}