import joblib
import numpy as np
from pathlib import Path

#сохраняем модель и эмеддинги 
def save_model(model_pipeline, game_embeddings, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'models'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / 'recsys_pipeline.pkl'
    embeddings_path = output_dir / 'game_embeddings.npy'

    joblib.dump(model_pipeline, pipeline_path)
    np.save(embeddings_path, game_embeddings)

    print(f'Модель сохранена: {pipeline_path}')
    print(f'Эмбеддинги сохранены: {embeddings_path}')

    return pipeline_path, embeddings_path

#загружаем модель и эмбеддинги 
def load_model(output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'models'

    output_dir = Path(output_dir)

    pipeline_path = output_dir / 'recsys_pipeline.pkl'
    embeddings_path = output_dir / 'game_embeddings.npy'

    model_pipeline = joblib.load(pipeline_path)
    game_embeddings = np.load(embeddings_path)

    print(f'Модель загружена из: {pipeline_path}')
    print(f'Эмбеддинги загружены из: {embeddings_path}')

    return model_pipeline, game_embeddings

if __name__ == '__main__':
    from src.data.load_data import load_steam_data
    from src.features.build_features import clean_data
    from src.models.recommend import build_feature_matrix, train_model

    df_raw = load_steam_data()
    df = clean_data(df_raw.copy())
    feature_matrix = build_feature_matrix(df)
    model_pipeline, game_embeddings = train_model(feature_matrix)

    save_model(model_pipeline, game_embeddings)
