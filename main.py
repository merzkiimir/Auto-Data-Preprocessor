import pandas as pd
from core import *


def interactive_fill_missing(df, target, task_type):
    while True:
        total_nans = df.isnull().sum().sum()
        print(f"Всего пропусков: {total_nans}")

        if total_nans == 0:
            print("Все пропуски устранены! Переходим к кодированию текста.")
            break

        print("Выберите способ обработки пропусков:")
        print("1. удалить все строки с пропусками")
        print("2. удалить столбцы с пропусками по параметрам")
        print("3. выделить пропуски в отдельный класс")
        print("4. заполнить модой по столбцу")
        print("5. выход")

        s = int(input())
        
        match s:
            case 1:
                df = drop_nan_rows(df)
                print("все строки с пропусками удалены")
            case 2:
                print("Введите процент и коэффициент корреляции")
                perc = float(input("процент: "))
                korr = float(input("коэффициент: "))
                perc2 = float(input("процент для ном.: "))
                df = drop_nan_columns_smart(df, target, task_type, perc, korr, perc2)
            case 3:
                df = fill_nan_with_new_class(df)
            case 4:
                df = fill_nan_smart_impute(df)
            case 5:
                break
            case _:
                continue
    return df


def handle_outliers_interactive(df, target, task_type):
    while True:
        print("Выберите способ обработки выбросов:")
        print("1. одномерная обработка")
        print("2. двумерная обработка")
        print("3. выход")
        s = int(input())
    
        match s:
            case 1:
                perc = float(input("введите отклонения в процентах (например, 1 для 1%): ")) / 100
                df = one_dimens(df, target, perc)                                
            case 2:
                cont_pct = float(input("Введите процент аномалий для удаления (например, 3 для 3%): ")) / 100
                df = two_dimens(df, target, cont_pct)
            case 3:
                break
            case _:
                continue
                
    # Интерактивный ввод параметров для удаления нерелевантных данных
    if task_type == "regression":
        corrind = float(input("Введите мин. коэфф. корреляции для чисел (например, 0.1): "))
        mi_ind = float(input("Введите мин. порог полезности (Mutual Info) для текста (например, 0.01): "))
    else:
        corrind = 0.1 # Заглушка, если классификация
        mi_ind = float(input("Введите мин. порог полезности (Mutual Info) (например, 0.01): "))
        
    df = zero_corr(df, target, task_type, corrind=corrind, mi_ind=mi_ind)
    
    # Интерактивный ввод для мультиколлинеарности
    threshold = float(input("Введите порог корреляции между признаками (например, 0.85): "))
    df = drop_highly_correlated_features(df, target, threshold=threshold)
    
    return df


def main():
    print("=== Умный пайплайн подготовки данных ===")
    file_name = input("Введите имя файла (например, train.csv): ")
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print("Файл не найден!")
        return

    target = input("Введите название целевой переменной (например, SalePrice): ")
    task_type = find_task_type(df, target)

    while True:
        print(f"\n--- Текущий статус ---")
        print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
        print(f"Пропусков осталось: {df.isnull().sum().sum()}")
        
        print("\n--- ГЛАВНОЕ МЕНЮ ---")
        print("0. Найти и преобразовать скрытые категории (Type Casting)")
        print("1. Очистить мусорные столбцы (Умный фильтр вариативности)")
        print("2. Обработать пропуски (Умное удаление + Заливка)")
        print("3. Найти и удалить выбросы, слабо коррелирующие стобцы и дубликаты")
        print("4. Закодировать текст и промасштабировать числа")
        print("5. Применить метод главных компонент (PCA)")
        print("6. Сохранить готовый датасет и выйти")
        
        choice = input("\nВаш выбор: ")
        
        match choice:
            case '0':
                df = convert_hidden_categories(df, target)
            case '1':
                print(df.head())
                print("Выберите номер столбца для удаления ID (если нет, введите 0):")
                cl = list(map(int, input().split()))
                df = drop_columns_by_index(df, cl) # Вызов переименованной функции из core.py
                
                df = parse_dt(df)
                
                om = float(input("Введите частоту самого популярного значения (0-100, например 90): ")) / 100
                n = float(input("Введите мин. разницу влияния на таргет (например 10 для 10%): ")) / 100
                df = drop_useless_columns_smartly(df, target, task_type, om, n)
            case '2':
                df = interactive_fill_missing(df, target, task_type)
            case '3':
                df = handle_outliers_interactive(df, target, task_type)
            case '4':
                print("Масштабировать бинарные (OHE) признаки тоже?")
                print("1 - Да, скейлить всё (Рекомендуется, если дальше будет PCA или линейная регрессия)")
                print("2 - Нет, скейлить только числа (Рекомендуется для деревьев решений)")
                ans = input("Ваш выбор: ")
                if ans == '1':
                    df = scale_and_encode_final(df, target, scale_all=True)
                else:
                    df = scale_and_encode_final(df, target, scale_all=False)
            case '5':
                d = float(input("Введите процент дисперсии для сохранения PCA (например, 95): ")) / 100
                df = ppccaa(df, target, d)
            case '6':
                df.to_csv(f"clean_{file_name}", index=False)
                print(f"Данные сохранены в 'clean_{file_name}'. До встречи!")
                break
            case _:
                print("Неизвестная команда, попробуйте еще раз.")

if __name__ == "__main__":
    main()