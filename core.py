import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA



def drop_columns_by_index(df, indices_list):
    #(1-based)
    if not indices_list or indices_list[0] == 0:
        return df
    
    indices_array = np.array(indices_list) - 1
    return df.drop(df.columns[indices_array], axis=1)


def parse_dt(df):
    print("--- Поиск и обработка дат ---")

    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in obj_cols:
        if df[col].notna().sum() == 0:
            continue
            
        sample = df[col].dropna().sample(min(100, len(df[col].dropna())))
        
        converted_sample = pd.to_datetime(sample, errors='coerce')
        
        if converted_sample.notna().mean() > 0.8:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Колонка '{col}' распознана как дата.")

    dt_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
    for col in dt_cols:
        loc = df.columns.get_loc(col)
        
        df.insert(loc, f"{col}_second", df[col].dt.second)
        df.insert(loc, f"{col}_minute", df[col].dt.minute)
        df.insert(loc, f"{col}_hour", df[col].dt.hour)
        df.insert(loc, f"{col}_day", df[col].dt.day)
        df.insert(loc, f"{col}_month", df[col].dt.month)
        df.insert(loc, f"{col}_year", df[col].dt.year)
        
        df.insert(loc, f"{col}_dayofweek", df[col].dt.dayofweek)
        
        df = df.drop(columns=[col])
        
    return df


def zero_corr(df, target, task_type, corrind = 0.1, mi_ind = 0.01):
    print("Удаление данных, которые не коррелируют")
    cols_to_del = []

    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    if task_type == "regression":
        nc = df.select_dtypes(include=['number']).columns
        num_cols = [col for col in nc if col != target]
        #corrind = float(input("Введите мин. коэфф. корреляции для чисел"))
        
        for col in num_cols:
            if abs(df[col].corr(df[target])) <= corrind:
                cols_to_del.append(col)
        
        cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        
        #mi_ind = float(input("Введите мин. порог полезности (Mutual Information) для текста (обычно 0.01 - 0.05): "))
        
        for col in cat_cols:
            encoded_col = pd.factorize(df[col])[0].reshape(-1, 1)
            
            mi_score = mutual_info_regression(encoded_col, df[target], random_state=42)[0]
            
            if mi_score <= mi_ind:
                cols_to_del.append(col)


    else: 
        #mi_ind = float(input("Введите мин. порог полезности (Mutual Information) (например 0.01): "))
    
        for col in df.columns:
            if col == target:
                continue
        
            if not pd.api.types.is_numeric_dtype(df[col]):
                X_col = pd.factorize(df[col])[0].reshape(-1, 1)
            else:
                X_col = df[col].values.reshape(-1, 1)
                
            mi_score = mutual_info_classif(X_col, y, random_state=42)[0]
            
            if mi_score <= mi_ind:
                cols_to_del.append(col)
        


    df = df.drop(columns=cols_to_del)
    print(f"Удалено нерелевантных столбцов: {len(cols_to_del)}")
    return df



def drop_highly_correlated_features(df, target, threshold = 0.85):
    print("\n--- Удаление дублирующих признаков (Мультиколлинеарность) ---")
     
    #threshold = float(input("Введите порог корреляции между признаками (например, 0.85): "))
        
    num_df = df.select_dtypes(include=['number'])
    
    if target in num_df.columns:
        num_df = num_df.drop(columns=[target])
        
    corr_matrix = num_df.corr().abs()
    
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    cols_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    
    df = df.drop(columns=cols_to_drop)
    print(f"Удалено {len(cols_to_drop)} дублирующих признаков.")
    return df



def drop_useless_columns_smartly(df, target, task_type, om = 90, n = 10):
    
    print("Удаление излишних данных (маленькая вариативность и отсутствие корреляции с целевой переменной)")
    #om = float(input("введите частосту (0-100):")) / 100
    #n = float(input("введите разницу (например 10%): ")) / 100

    cols_to_drop = []

    for col in df.columns:

        if col == target:
            continue

        if df[col].dropna().empty:
            continue
        
        name = df[col].value_counts(normalize=True).index[0]
        omt = df[col].value_counts(normalize=True).iloc[0]

        if omt>= om:

            gr1 = df[df[col] == name]
            gr2 = df[df[col] != name]

            if gr2.empty:
                cols_to_drop.append(col)
                continue

            if task_type == 'regression':
                mean_1 = gr1[target].mean()
                mean_2 = gr2[target].mean()
                
                
                if mean_1 == 0:
                    diff = abs(mean_1 - mean_2)
                else:
                    diff = abs(mean_1 - mean_2) / abs(mean_1)


            elif task_type == 'classification':
                
                top_target_class = df[target].value_counts().index[0]
                
                
                dist_popular = gr1[target].value_counts(normalize=True).get(top_target_class, 0.0)
                dist_rare = gr2[target].value_counts(normalize=True).get(top_target_class, 0.0)
                
                diff = abs(dist_popular - dist_rare)
            

            if diff < n:
                cols_to_drop.append(col)
    print(f"Найдено и удалено бесполезных столбцов: {len(cols_to_drop)}")

    return df.drop(columns=cols_to_drop)


def drop_nan_rows(df):
    df = df.dropna()
    return df


def drop_nan_columns_smart(df, target, task_type, perc, korr, perc2):
    col_todrop = []
    
    for col in df.columns:

        if col == target:
            continue

        nan_pr =  df[col].isnull().sum() / len(df)

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if nan_pr >= perc and is_numeric and task_type == "regression":
            if abs(df[col].corr(df[target])) <= korr:
                col_todrop.append(col)
        
        elif nan_pr >= perc2 and (not is_numeric or task_type == "classification"):
            col_todrop.append(col)

    df = df.drop(columns=col_todrop)
    print(f"Удалено столбцов: {len(col_todrop)}")
    return df

def fill_nan_with_new_class(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('None')
    print("пропуски выделены")
    return df

def fill_nan_smart_impute(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])


    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())


    date_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
    for col in date_cols:
        df[col] = df[col].ffill()

    
    print("Все столбцы заполнены (мода/медиана/ffill).")
    return df



def one_dimens(df, target, perc = 0.01):
    nc = df.select_dtypes(include=['number']).columns
    num_cols = [col for col in nc if col != target]
    start_rows = len(df)

    #perc = float(input("введите отклонения в процентах: ")) / 100
    mask = pd.Series(True, index=df.index)

    for col in num_cols:
        lower_bound = df[col].quantile(perc)
        upper_bound = df[col].quantile(1 - perc)
        
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
        
    df = df[mask]
    print(f"Отсечение по {perc*100}% краям удалило {start_rows - len(df)} строк.")
    return df

def two_dimens(df, target, cont_pct = 3):
    nc = df.select_dtypes(include=['number']).columns
    num_cols = [col for col in nc if col != target]
    start_rows = len(df)

    #cont_pct = float(input("Введите процент аномалий для удаления (например, 3 для 3%): ")) / 100
    iso = IsolationForest(contamination=cont_pct, random_state=42)
    outlier_labels = iso.fit_predict(df[num_cols])
    df = df[outlier_labels == 1]
    print(f"Изолирующий лес нашел и удалил {start_rows - len(df)} строк.")
    return df

def scale_and_encode_final(df, target, scale_all=True):
    print("\n--- Кодирование и масштабирование ---")
    
    y = df[target].copy()
    df = df.drop(columns=[target])
    
    if scale_all:
        #1 Сначала OHE, потом скейл
        df = pd.get_dummies(df, drop_first=True)
        print("Текстовые признаки превращены в бинарные (One-Hot Encoding).")
        
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        df = df_scaled
        print("Отмасштабированы все колонки.")
        
    else:
        #2 cкейл числа, потом OHE
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
            for col in num_cols:
                df[col] = df_scaled[col]
            print(f"Отмасштабировано {len(num_cols)} числовых колонок.")
            
        df = pd.get_dummies(df, drop_first=True)
        print("Текстовые признаки превращены в бинарные (One-Hot Encoding).")
        
    df[target] = y
    print(f"\nГотово! Итоговый размер таблицы: {df.shape[0]} строк, {df.shape[1]} колонок.")
    return df


def ppccaa(df, target, d = 95):
    print("\n--- Метод главных компонент (PCA) ---")
    
    features = df.drop(columns=[target], errors='ignore')
    if features.select_dtypes(include=['object', 'datetime']).shape[1] > 0:
        print("ОШИБКА: В таблице остались текстовые признаки или даты!")
        print("Сначала выполните Шаг 4 (Кодировка и масштабирование).")
        return df
        
    #s = float(input("Введите процент полезной информации (дисперсии) для сохранения (например, 95): ")) / 100
        
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Признаков до сжатия: {X.shape[1]}")
    
    pca = PCA(n_components=d, random_state=42) 
    X_pca = pca.fit_transform(X)
    
    pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    
    df_new = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    df_new[target] = y
    
    print(f"Осталось главных компонент после PCA: {X_pca.shape[1]}")
    
    return df_new

def convert_hidden_categories(df, target):
    print("\n--- Поиск скрытых категорий ---")
    
    num_cols = df.select_dtypes(include=['number']).columns
    converted_count = 0
    
    for col in num_cols:
        if col == target:
            continue
            
        unique_vals = df[col].nunique()
        
        if unique_vals < 10:
            df[col] = df[col].astype(str)
            converted_count += 1
                
    print(f"\nИтого преобразовано столбцов в текст: {converted_count}")
    return df

def find_task_type(df, target):
    if target not in df.columns:
        print("Целевая переменная не найдена. Работа остановлена.")
        return 'no'

    if df[target].dtype.kind in ['O', 'b'] or df[target].nunique() < 10:
        task_type = 'classification'
        print(f"Целевая переменная '{target}' определена как КЛАССИФИКАЦИЯ.")
    else:
        task_type = 'regression'
        print(f"Целевая переменная '{target}' определена как РЕГРЕССИЯ.")
    return task_type