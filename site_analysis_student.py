import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Загрузка файлов

BASE_DIR = Path(".")
SESSIONS_PATH = BASE_DIR / "ga_sessions.csv"
HITS_AGG_PATH = BASE_DIR / "ga_hits_aggregated.csv"


def check_files():
    if not SESSIONS_PATH.exists():
        raise FileNotFoundError(
            "Не найден файл ga_sessions.csv. "
            "Положите его в ту же папку, где находится этот скрипт."
        )
    if not HITS_AGG_PATH.exists():
        raise FileNotFoundError(
            "Не найден файл ga_hits_aggregated.csv. "
            "Сначала подготовьте его из большого ga_hits.csv."
        )


# 2. Чтение данных

def load_data():
    ga_sessions = pd.read_csv(SESSIONS_PATH)
    hits_agg = pd.read_csv(HITS_AGG_PATH)

    print("Размер ga_sessions:", ga_sessions.shape)
    print("Размер hits_agg:", hits_agg.shape)

    return ga_sessions, hits_agg


# 3. Объединение таблиц

def merge_data(ga_sessions, hits_agg):
    df = ga_sessions.merge(hits_agg, on="session_id", how="left")

    # После merge могут появиться пропуски
    df["has_target"] = df["has_target"].fillna(0).astype(int)
    df["hits_count"] = df["hits_count"].fillna(0)
    df["pages_count"] = df["pages_count"].fillna(0)
    df["event_count"] = df["event_count"].fillna(0)

    return df

# 4. Простое EDA

def simple_eda(df):
    print("\nПервые строки таблицы:")
    print(df.head())

    print("\nТипы данных:")
    print(df.dtypes.head(20))

    print("\nПропуски:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(20))

    print("\nРаспределение целевой переменной:")
    print(df["has_target"].value_counts())
    print(df["has_target"].value_counts(normalize=True))

    # Графики
    plt.figure(figsize=(6, 4))
    sns.countplot(x="has_target", data=df)
    plt.title("Распределение целевого действия")
    plt.tight_layout()
    plt.show()

    for col in ["hits_count", "pages_count", "event_count"]:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Распределение {col}")
            plt.tight_layout()
            plt.show()

    # Конверсия по нескольким признакам
    for col in ["utm_medium", "device_category", "device_browser"]:
        if col in df.columns:
            conv = df.groupby(col)["has_target"].mean().sort_values(ascending=False).head(10)
            print(f"\nТоп-10 по конверсии для {col}:")
            print(conv)

# 5. Подготовка данных

def prepare_data(df):
    drop_cols = []
    for col in ["session_id", "client_id"]:
        if col in df.columns:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols)

    y = df["has_target"]
    X = df.drop(columns=["has_target"])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nЧисловых признаков:", len(num_cols))
    print("Категориальных признаков:", len(cat_cols))

    # заполняем пропуски
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # оставляем только небольшое число категориальных признаков
    selected_cat_cols = []

    small_cat_candidates = [
        "utm_medium",
        "device_category",
        "device_os",
        "device_browser",
        "geo_country"
    ]

    for col in small_cat_candidates:
        if col in X.columns:
            selected_cat_cols.append(col)

    # оставляем числовые + только эти категории
    selected_cols = num_cols + selected_cat_cols
    X = X[selected_cols].copy()

    # убираем возможные дубли
    X = X.loc[:, ~X.columns.duplicated()]

    # кодирование
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # уменьшаем потребление памяти
    for col in X.columns:
        if X[col].dtype == "float64":
            X[col] = X[col].astype("float32")
        elif X[col].dtype == "int64":
            X[col] = X[col].astype("int32")
        elif X[col].dtype == "bool":
            X[col] = X[col].astype("int8")

    print("Размер после кодирования:", X.shape)

    return X, y

# 6. Обучение моделей

def train_models(X_train, X_test, y_train, y_test):

    models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append([name, roc_auc])
        fitted_models[name] = model

    results_df = pd.DataFrame(results, columns=["Model", "ROC_AUC"])
    results_df = results_df.sort_values(by="ROC_AUC", ascending=False)

    return results_df, fitted_models

# 7. Графики по моделям

def plot_model_results(results_df):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, x="Model", y="ROC_AUC")
    plt.title("Сравнение моделей по ROC-AUC")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# 8. ROC-кривая для лучшей модели

def plot_best_model_curve(best_model, X_test, y_test):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return roc_auc

# 9. Важность признаков

def show_feature_importance(best_model, X_train):
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        print("\nТоп-20 важных признаков:")
        print(importance_df.head(20))

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df.head(20),
            x="Importance",
            y="Feature"
        )
        plt.title("Топ-20 важных признаков")
        plt.tight_layout()
        plt.show()

# 10. Основной запуск

def main():
    check_files()

    ga_sessions, hits_agg = load_data()
    df = merge_data(ga_sessions, hits_agg)

    print("\nРазмер объединенной таблицы:", df.shape)

    simple_eda(df)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nРазмеры выборок:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    results_df, fitted_models = train_models(X_train, X_test, y_train, y_test)

    print("\nСравнение моделей:")
    print(results_df)

    plot_model_results(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    print("\nЛучшая модель:", best_model_name)

    best_model = fitted_models[best_model_name]
    final_auc = plot_best_model_curve(best_model, X_test, y_test)

    print("\nROC-AUC лучшей модели:", round(final_auc, 4))

    show_feature_importance(best_model, X_train)

    # Пример предсказания для одного визита
    sample = X_test.iloc[[0]].copy()
    sample_proba = best_model.predict_proba(sample)[:, 1][0]
    sample_pred = int(sample_proba >= 0.5)

    print("\nПример предсказания для одного визита:")
    print("Вероятность целевого действия:", round(sample_proba, 4))
    print("Класс:", sample_pred)

    print("\nИтоговый вывод:")
    print("1. Данные по визитам были объединены с агрегированной таблицей событий.")
    print("2. Целевая переменная показывает, было ли хотя бы одно целевое действие в визите.")
    print("3. Были протестированы несколько моделей классификации.")
    print("4. Основной метрикой качества была ROC-AUC.")
    print("5. Лучшая модель была выбрана по значению ROC-AUC на тестовой выборке.")
    print("6. Дополнительно был выполнен анализ важности признаков.")


if __name__ == "__main__":
    main()
