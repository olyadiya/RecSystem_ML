import pandas as pd
from pathlib import Path

def load_steam_data():
    """Загружает сырые данные из data/raw/steam.csv"""
    
    root = Path(__file__).parent.parent.parent
    file_path = root / 'data' / 'raw' / 'steam.csv'
    return pd.read_csv(file_path)
