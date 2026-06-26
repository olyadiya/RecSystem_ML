#src/models/recommend.py

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MultiLabelBinarizer, StandardScaler)

#собираем матрицу признаков по очищенному датафрейму 
def build_feature_matrix(df):
    def make_binary(df, col, prefix):
        mlb = MultiLabelBinarizer()
        binary = mlb.fit_transform(df[col].str.lower().str.split(';'))
        return pd.DataFrame(binary, columns=[f'{prefix}_{c}' for c in mlb.classes_])

    genres_df = make_binary(df, 'genres', 'genre')
    categories_df = make_binary(df, 'categories', 'cat')
    tags_df = make_binary(df, 'steamspy_tags', 'tag')
    platforms_df = make_binary(df, 'platforms', 'platform')

    #Убираем из тегов всё, что есть в других колонках
    cols_to_remove = set()
    cols_to_remove.update([c.replace('genre_', '') for c in genres_df.columns])
    cols_to_remove.update([c.replace('cat_', '') for c in categories_df.columns])
    cols_to_remove.update([c.replace('platform_', '') for c in platforms_df.columns])
    tags_clean = tags_df.drop(
        columns=[f'tag_{v}' for v in cols_to_remove if f'tag_{v}' in tags_df.columns],
        errors='ignore'
    )
    
#отбираем признаки для модели 
    binary_cols_df = pd.concat([genres_df, categories_df, tags_clean, platforms_df], axis=1)
    numeric_cols = ['rating_ratio', 'positive_ratings', 'negative_ratings', 'price', 'developer_encoded']

#нормализуем числовые признаки и собираме матрицу
    scaler_pre = StandardScaler()
    numeric_scaled = scaler_pre.fit_transform(df[numeric_cols])
    df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_cols, index=df.index)
    feature_matrix = pd.concat([df_numeric, binary_cols_df], axis=1)
    return feature_matrix

#Обучение модели 
#Подбираем число компонент по порогу дисперсии
def train_model(feature_matrix, variance_threshold=0.90):
    svd_analysis = TruncatedSVD(n_components=200, random_state=42)
    svd_analysis.fit(feature_matrix.values)
    cumvar = np.cumsum(svd_analysis.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= variance_threshold)) + 1

    print(f'Компонент для {variance_threshold:.0%} дисперсии: {n_components}')

    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svd', TruncatedSVD(n_components=n_components, random_state=42))
    ])

    embeddings = model_pipeline.fit_transform(feature_matrix.values)
    total_variance = model_pipeline.named_steps['svd'].explained_variance_ratio_.sum()
    print(f'Сохранено дисперсии: {total_variance:.1%}')
    print(f'Матрица эмбеддингов: {embeddings.shape}')

    return model_pipeline, embeddings

#строим матрицу косинусного сходства
def build_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

#создаем рекомендации
def get_recommendations(game_name, df, similarity_matrix, n=5):
    matches = df[df['name'].str.lower() == game_name.lower()]
    if matches.empty:
        matches = df[df['name'].str.lower().str.contains(game_name.lower(), na=False)]
        if matches.empty:
            print(f'Игра "{game_name}" не найдена в датасете')
            return None
        print(f'Используем: "{matches.iloc[0]["name"]}"')

    game_idx = matches.index[0]
    sim_scores = sorted(enumerate(similarity_matrix[game_idx]), key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != game_idx][:n]

    top_indices = [i for i, _ in sim_scores]
    top_scores = [s for _, s in sim_scores]

    result = df.loc[top_indices, ['name', 'genres', 'categories', 'platforms', 'rating_ratio', 'price']].copy()
    result['similarity_score'] = [round(s, 4) for s in top_scores]
    result['rating_ratio'] = result['rating_ratio'].round(2)
    return result.reset_index(drop=True)

#для локального запуска
if __name__ == '__main__':
    from pathlib import Path
    from src.data.load_data import load_steam_data
    from src.features.build_features import clean_data

    df_raw = load_steam_data()
    df = clean_data(df_raw.copy())

    feature_matrix = build_feature_matrix(df)
    model_pipeline, embeddings = train_model(feature_matrix)
    similarity_matrix = build_similarity_matrix(embeddings)

    #Пример
    print(get_recommendations('Counter-Strike', df, similarity_matrix))
