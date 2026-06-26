#Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

#загрузка и очистка данных 
df_raw = pd.read_csv('/content/steam.csv')
def clean_data(df):

    # 1. ОЧИСТКА
    df['developer'] = df['developer'].fillna('Unknown')
    df['publisher'] = df['publisher'].fillna('Unknown')

    # 2. НОВЫЕ ПРИЗНАКИ

    # release_year - год релиза игры (извлечён из release_date)
    df['release_year'] = df['release_date'].str[:4].astype(int)

    # rating_ratio - доля положительных отзывов у игры, посчитано через positive_ratings и negative_ratings
    df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
    df['rating_ratio'] = df['positive_ratings'] / (df['total_ratings'] + 1)

    def parse_owners(x):
        try:
            low, high = x.split('-')
            return (int(low.replace(',', '')) + int(high.replace(',', ''))) / 2
        except:
            return None

    # owners_mean - среднее количество обладателей игры, извлечено из диапазона в owners
    df['owners_mean'] = df['owners'].apply(parse_owners)


    # 3. БИНАРНЫЕ ПРИЗНАКИ (для One-Hot)

    def make_binary(df, col, prefix):
        mlb = MultiLabelBinarizer()
        binary = mlb.fit_transform(df[col].str.lower().str.split(';'))
        return pd.DataFrame(binary, columns=[f'{prefix}_{c}' for c in mlb.classes_])

    genres_df = make_binary(df, 'genres', 'genre')
    categories_df = make_binary(df, 'categories', 'cat')
    tags_df = make_binary(df, 'steamspy_tags', 'tag')
    platforms_df = make_binary(df, 'platforms', 'platform')

    # Убираем из тегов всё, что есть в других колонках
    cols_to_remove = set()
    cols_to_remove.update([col.replace('genre_', '') for col in genres_df.columns])
    cols_to_remove.update([col.replace('cat_', '') for col in categories_df.columns])
    cols_to_remove.update([col.replace('platform_', '') for col in platforms_df.columns])
    tags_clean = tags_df.drop(
        columns=[f'tag_{val}' for val in cols_to_remove if f'tag_{val}' in tags_df.columns],
        errors='ignore'
    )


    # 4. ТЕКСТОВЫЕ ПРИЗНАКИ (для TF-IDF)

    df['genres_text'] = df['genres'].str.lower().str.replace(';', ' ', regex=False)
    df['categories_text'] = df['categories'].str.lower().str.replace(';', ' ', regex=False)
    df['platforms_text'] = df['platforms'].str.lower().str.replace(';', ' ', regex=False)

    # Очищаем теги от жанров и категорий для текста
    all_genres = set()
    for val in df['genres'].dropna():
        if isinstance(val, str):
            for v in val.split(';'):
                all_genres.add(v.strip().lower())

    all_categories = set()
    for val in df['categories'].dropna():
        if isinstance(val, str):
            for v in val.split(';'):
                all_categories.add(v.strip().lower())

    def clean_tags_text(tags_str):
        if not isinstance(tags_str, str):
            return ''
        tags = [t.strip().lower() for t in tags_str.split(';')]
        filtered = [t for t in tags if t not in all_genres and t not in all_categories]
        return ' '.join(filtered)

    df['tags_text_clean'] = df['steamspy_tags'].apply(clean_tags_text)

    # 5. КОДИРОВАНИЕ developer
    le = LabelEncoder()
    df['developer_encoded'] = le.fit_transform(df['developer'])

    # 6. ОБЪЕДИНЯЕМ One-Hot признаки в основной DataFrame
    df_onehot = pd.concat([genres_df, categories_df, tags_clean, platforms_df], axis=1)
    df = pd.concat([df, df_onehot], axis=1)

    return df

df = clean_data(df_raw.copy())

df.head(3)

#отбор признаков для модели 
binary_cols = [col for col in df.columns if col.startswith(('genre_', 'cat_', 'tag_', 'platform_'))]
numeric_cols = ['rating_ratio', 'positive_ratings', 'negative_ratings', 'price', 'developer_encoded']

print(f'Бинарных признаков:  {len(binary_cols)}')
print(f'Числовых признаков:  {len(numeric_cols)}')
print(f'Итого признаков:     {len(binary_cols) + len(numeric_cols)}')

##Нормализуем числовые признаки вручную
scaler_pre = StandardScaler()
numeric_scaled = scaler_pre.fit_transform(df[numeric_cols])
df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_cols, index=df.index)

# Собираем итоговую матрицу: числовые (нормализованные) + бинарные
feature_matrix = pd.concat([df_numeric, df[binary_cols]], axis=1)

print(f'Размер матрицы признаков до SVD: {feature_matrix.shape}')
print(f'  → {feature_matrix.shape[0]} игр × {feature_matrix.shape[1]} признаков')

# Обучаем SVD с максимальным числом компонент, чтобы посмотреть на дисперсию
svd_analysis = TruncatedSVD(n_components=200, random_state=42)
svd_analysis.fit(feature_matrix.values)

cumvar = np.cumsum(svd_analysis.explained_variance_ratio_)

#Находим минимальное число компонент для порогов 80% и 90%
n_80 = np.argmax(cumvar >= 0.80) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f'Для 80% дисперсии нужно компонент: {n_80}')
print(f'Для 90% дисперсии нужно компонент: {n_90}')

#Обучение модели TruncatedSVD
N_COMPONENTS = n_90

model_pipeline = Pipeline([
    #нормализуем всю матрицу перед SVD
    ('scaler', StandardScaler()),
    #обучение
    ('svd', TruncatedSVD(n_components=N_COMPONENTS, random_state=42))
])

game_embeddings = model_pipeline.fit_transform(feature_matrix.values)

print(f'Матрица эмбеддингов игр: {game_embeddings.shape}')
print(f'  → {game_embeddings.shape[0]} игр × {game_embeddings.shape[1]} латентных факторов')
print(f'\nСжатие: {feature_matrix.shape[1]} признаков → {N_COMPONENTS} латентных факторов')

total_variance = model_pipeline.named_steps['svd'].explained_variance_ratio_.sum()
print(f'Сохранено дисперсии: {total_variance:.1%}')

#Построение матрицы косинусного сходства
similarity_matrix = cosine_similarity(game_embeddings)

print(f'Матрица сходства: {similarity_matrix.shape}')

#проверка - ожидается 1.0
print(f'{np.diag(similarity_matrix).mean():.4f}')

#функция рекомендаций 
def get_recommendations(game_name, df, similarity_matrix, n=5):
    #Поиск игры
    matches = df[df['name'].str.lower() == game_name.lower()]

    if matches.empty:
        #частичное совпадение, если точного нет
        matches = df[df['name'].str.lower().str.contains(game_name.lower(), na=False)]
        if matches.empty:
            print(f'игра "{game_name}" не найдена в датасете')
            return None
        print(f'используем: "{matches.iloc[0]["name"]}"')

    game_idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[game_idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != game_idx]

    top_n = sim_scores[:n]

    top_indices = [i for i, _ in top_n]
    top_scores  = [s for _, s in top_n]

    result = df.loc[top_indices, ['name', 'genres', 'categories', 'platforms', 'rating_ratio', 'price']].copy()
    result['similarity_score'] = [round(s, 4) for s in top_scores]
    result['rating_ratio'] = result['rating_ratio'].round(2)

    return result.reset_index(drop=True)
