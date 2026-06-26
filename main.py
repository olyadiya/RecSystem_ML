from src.data.load_data import load_steam_data
from src.features.build_features import clean_data

def main():
    # 1. Загружаем сырые данные
    df = load_steam_data()
    
    # 2. Чистим и создаём признаки
    df_clean = clean_data(df)
    
    # 3. Сохраняем результат
    df_clean.to_csv('data/processed/steam_clean.csv', index=False)

    
    print("Данные загружены, очищены, добавлены новые признаки.")
    print("Чистые данные сохранены в steam_clean.csv")

if __name__ == "__main__":
    main()
