from src.data.load_data import load_steam_data
from src.features.build_features import clean_data
from src.models.recommend import build_feature_matrix, train_model, build_similarity_matrix, get_recommendations
from src.models.download_model import save_model


def main():
    #Загружаем сырые данные
    print("Загрузка данных...")
    df = load_steam_data()

    #Чистим и создаём признаки
    print("Очистка и подготовка признаков...")
    df_clean = clean_data(df)
    df_clean.to_csv('data/processed/steam_clean.csv', index=False)
    print("Чистые данные сохранены в data/processed/steam_clean.csv")

    #Строим матрицу признаков
    print("Построение матрицы признаков...")
    feature_matrix = build_feature_matrix(df_clean)

    #Обучаем модель
    print("Обучение модели...")
    model_pipeline, embeddings = train_model(feature_matrix)

    #Строим матрицу сходства
    print("Построение матрицы сходства...")
    similarity_matrix = build_similarity_matrix(embeddings)

    #Сохраняем модель
    print("Сохранение модели...")
    save_model(model_pipeline, embeddings)

    #Пример рекомендаций
    print("Пример рекомендаций:")
    result = get_recommendations('Counter-Strike', df_clean, similarity_matrix)
    if result is not None:
        print(result)

    print("\nГотово! Модель обучена и сохранена.")

if __name__ == "__main__":
    main()
