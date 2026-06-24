import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def clean_data(df):
    """
    Очищает данные и создаёт новые признаки.
    Возвращает очищенный DataFrame.
    """
    
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
