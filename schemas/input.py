from pydantic import BaseModel

class MoleculeInput(BaseModel):
    smiles: str