import numpy as np

# SAME mapping must match training
chars = list("CNOSP123456789=#()[]")
char_to_int = {c: i + 1 for i, c in enumerate(chars)}

MAX_LEN = 50  # same as training (IMPORTANT)

def encode_smiles(smiles):
    encoded = []

    for c in smiles:
        encoded.append(char_to_int.get(c, 0))

    # padding
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return np.array(encoded).reshape(1, MAX_LEN)