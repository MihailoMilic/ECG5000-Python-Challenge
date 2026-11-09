from pathlib import Path
import numpy as np

def load_training_data():
    path = Path(__file__).parent / "ECG5000" / "ECG5000_TRAIN.txt"
    return np.loadtxt(path)