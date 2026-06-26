import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from src.features.build_features import clean_data

#загрузка данных
def load_data():
    try:
        import kagglehub
        import os
        path = kagglehub.dataset_download("nikdavis/steam-store-games")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

        if 'steam.csv' in csv_files:
            df = pd.read_csv(os.path.join(path, 'steam.csv'))
        else:
            for f in csv_files:
                test_df = pd.read_csv(os.path.join(path, f))
                if 'developer' in test_df.columns and 'genres' in test_df.columns:
                    df = test_df
                    break

    except ImportError:
        #Локальный fallback
        from src.data.load_data import load_steam_data
        df = load_steam_data()

    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    return df

#делим данные на основную и тестовую выборки 
def split_data(df, test_size=0.2, random_state=42):
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"Train: {len(train_df)} игр, Test: {len(test_df)} игр")
    return train_df, test_df

#обучаем модель на тестовой 
def train_model(train_df, n_components=29):
    binary_cols = [col for col in train_df.columns if col.startswith(('genre_', 'cat_', 'tag_', 'platform_'))]
    numeric_cols = ['rating_ratio', 'positive_ratings', 'negative_ratings', 'price', 'developer_encoded']

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(train_df[numeric_cols])
    df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_cols, index=train_df.index)
    feature_matrix = pd.concat([df_numeric, train_df[binary_cols]], axis=1)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(feature_matrix.values)

    print(f"Модель обучена. Эмбеддинги: {embeddings.shape}")
    return scaler, svd, embeddings, binary_cols, numeric_cols

#Преобразуем test-выборку обученной моделью 
def transform_test(test_df, scaler, svd, binary_cols, numeric_cols):
    numeric_scaled = scaler.transform(test_df[numeric_cols])
    df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_cols, index=test_df.index)
    feature_matrix = pd.concat([df_numeric, test_df[binary_cols]], axis=1)
    embeddings = svd.transform(feature_matrix.values)
    print(f"Test преобразован: {embeddings.shape}")
    return embeddings

#Находит для test-игры похожие train-игры
def get_recommendations_for_test(game_name, train_df, test_df, train_similarity, n=5):
    matches = test_df[test_df['name'].str.lower() == game_name.lower()]
    if matches.empty:
        matches = test_df[test_df['name'].str.lower().str.contains(game_name.lower(), na=False)]
        if matches.empty:
            return None

    test_idx = matches.index[0]
    test_pos = test_df.index.get_loc(test_idx)
    sim_row = train_similarity[test_pos % len(train_df)]
    top_indices = np.argsort(sim_row)[::-1][1:n + 1]
    top_scores = sim_row[top_indices]
    top_games = train_df.iloc[top_indices]['name'].values

    return top_games, top_scores

#считаем precision и срдний коэфициент сходства
def evaluate(train_df, test_df, train_similarity, sample_size=200, n=5):
    precision_scores = []
    similarity_scores = []

    sample_test = test_df.head(sample_size)

    for idx, row in sample_test.iterrows():
        game_name = row['name']
        game_genres = set(str(row['genres']).split(';')) if pd.notna(row['genres']) else set()

        recs = get_recommendations_for_test(game_name, train_df, test_df, train_similarity, n=n)
        if recs is None:
            continue

        rec_games, rec_scores = recs
        similarity_scores.extend(rec_scores)

        relevant_count = 0
        for rec_game in rec_games:
            rec_row = train_df[train_df['name'] == rec_game]
            if not rec_row.empty:
                rec_genres = set(str(rec_row.iloc[0]['genres']).split(';')) if pd.notna(rec_row.iloc[0]['genres']) else set()
                if game_genres.intersection(rec_genres):
                    relevant_count += 1

        precision_scores.append(relevant_count / n)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

    print(f"\nPrecision@{n}: {avg_precision:.3f}")
    print(f"Средний коэффициент сходства: {avg_similarity:.3f}")
    print(f"В среднем {avg_precision * 100:.1f}% рекомендаций имеют общий жанр с исходной игрой.")

    return precision_scores, similarity_scores, avg_precision, avg_similarity

#строим и сохраняем графики оценки
def plot_results(precision_scores, similarity_scores, avg_precision, avg_similarity, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'reports' / 'figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(precision_scores, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=avg_precision, color='red', linestyle='--', linewidth=2, label=f'Среднее: {avg_precision:.3f}')
    ax1.set_xlabel('Precision@5')
    ax1.set_ylabel('Количество игр')
    ax1.set_title('Распределение Precision@5 на тестовой выборке')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(similarity_scores, bins=20, color='green', edgecolor='black', alpha=0.7)
    ax2.axvline(x=avg_similarity, color='red', linestyle='--', linewidth=2, label=f'Среднее: {avg_similarity:.3f}')
    ax2.set_xlabel('Коэффициент сходства')
    ax2.set_ylabel('Количество рекомендаций')
    ax2.set_title('Распределение коэффициентов сходства')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'train_test_evaluation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"График сохранён: {save_path}")

 #Сохраняем метрики в csv
def save_results(precision_scores, similarity_scores, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'reports'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        'precision': precision_scores,
        'similarity': similarity_scores[:len(precision_scores)]
    })
    save_path = output_dir / 'test_results.csv'
    results_df.to_csv(save_path, index=False)
    print(f"Результаты сохранены: {save_path}")


def print_examples(train_df, test_df, train_similarity, examples=None):
    if examples is None:
        examples = ['Counter-Strike', 'Stardew Valley', 'DOOM', 'Portal 2']

    print("\nПримеры рекомендаций (test → train):")
    for game in examples:
        recs = get_recommendations_for_test(game, train_df, test_df, train_similarity, n=3)
        if recs is not None:
            rec_games, rec_scores = recs
            print(f"\n'{game}':")
            for i, (g, s) in enumerate(zip(rec_games, rec_scores)):
                print(f"   {i + 1}. {g} (сходство: {s:.3f})")
        else:
            print(f"\nИгра '{game}' не найдена в test-выборке")


if __name__ == '__main__':
    print("=" * 60)
    print("ОЦЕНКА КАЧЕСТВА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    df_raw = load_data()
    df = clean_data(df_raw.copy())
    train_df, test_df = split_data(df)

    scaler, svd, train_embeddings, binary_cols, numeric_cols = train_model(train_df)
    test_embeddings = transform_test(test_df, scaler, svd, binary_cols, numeric_cols)

    train_similarity = cosine_similarity(train_embeddings)

    precision_scores, similarity_scores, avg_precision, avg_similarity = evaluate(
        train_df, test_df, train_similarity
    )

    print_examples(train_df, test_df, train_similarity)
    plot_results(precision_scores, similarity_scores, avg_precision, avg_similarity)
    save_results(precision_scores, similarity_scores)

    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    print(f"Train: {len(train_df)} игр | Test: {len(test_df)} игр")
    print(f"Precision@5: {avg_precision:.3f} ({avg_precision * 100:.1f}% рекомендаций имеют общий жанр)")
    print(f"Средний коэффициент сходства: {avg_similarity:.3f}")
    if avg_precision > 0.5:
        print("Качество рекомендаций хорошее (Precision@5 > 0.5)")
    else:
        print("Качество рекомендаций требует улучшения (Precision@5 < 0.5)")
    print("=" * 60)
