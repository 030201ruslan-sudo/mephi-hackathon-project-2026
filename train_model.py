# Отключаем предупреждения, чтобы вывод в консоли был чище
import warnings
warnings.filterwarnings("ignore")

# Path нужен для удобной и безопасной работы с путями к файлам
from pathlib import Path

# Основные библиотеки для работы с данными и сохранения модели
import joblib
import numpy as np
import pandas as pd

# Инструменты sklearn для предобработки и пайплайнов
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Изученные модели классификации
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Метрики качества
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Папка, в которой лежит текущий скрипт
BASE_DIR = Path(__file__).resolve().parent

# Пути к входным и выходным файлам
SESSIONS_FILE = BASE_DIR / "ga_sessions.csv"
TARGET_FILE = BASE_DIR / "session_target.csv"
MODEL_OUTPUT = BASE_DIR / "best_conversion_model.joblib"
REPORT_OUTPUT = BASE_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_OUTPUT = BASE_DIR / "feature_importance.csv"

# Сколько самых частых значений оставлять для категориальных признаков.
# Все остальные редкие значения будут объединяться в "other".
GROUP_TOP_N = {
    "utm_source": 12,
    "utm_medium": 8,
    "utm_campaign": 15,
    "device_browser": 10,
    "geo_city": 20,
    "geo_country": 10,
}

# Финальный набор признаков, на которых обучается модель
FINAL_FEATURES = [
    "visit_number",
    "log_visit_number",
    "visit_hour",
    "visit_dayofweek",
    "visit_month",
    "is_weekend",
    "is_organic",
    "is_paid",
    "is_first_visit",
    "is_returning",
    "is_night",
    "is_work_hours",
    "is_mobile",
    "is_desktop",
    "device_category",
    "device_os",
    "device_browser_group",
    "geo_country_group",
    "geo_city_group",
    "utm_source_group",
    "utm_medium_group",
    "utm_campaign_group",
]


def load_data() -> pd.DataFrame:
    """
    Загружает данные сессий и target.

    Важно:
    - ga_sessions.csv используется как источник признаков
    - session_target.csv содержит целевую переменную has_target,
      которую мы заранее построили из ga_hits.csv
    """
    if not SESSIONS_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {SESSIONS_FILE}")
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {TARGET_FILE}")

    sessions = pd.read_csv(SESSIONS_FILE, low_memory=False)
    target = pd.read_csv(TARGET_FILE)

    # Объединяем по session_id
    df = sessions.merge(target, on="session_id", how="left")

    # Если target для сессии не найден, считаем, что целевого действия не было
    df["has_target"] = df["has_target"].fillna(0).astype(int)

    return df


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт базовые безопасные признаки, которые известны на момент визита пользователя.

    Здесь не используются агрегаты из ga_hits.csv, чтобы избежать утечки данных.
    """
    df = df.copy()

    # Приводим дату и время к удобному виду
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    df["visit_time"] = df["visit_time"].astype(str)

    # Собираем полный datetime визита
    df["visit_datetime"] = pd.to_datetime(
        df["visit_date"].astype(str) + " " + df["visit_time"].astype(str),
        errors="coerce"
    )

    # Признаки времени визита
    df["visit_hour"] = df["visit_datetime"].dt.hour
    df["visit_dayofweek"] = df["visit_datetime"].dt.dayofweek
    df["visit_month"] = df["visit_datetime"].dt.month
    df["is_weekend"] = df["visit_dayofweek"].isin([5, 6]).astype(int)

    # Бинарные признаки по типу трафика
    df["is_organic"] = df["utm_medium"].isin(["organic", "referral", "(none)"]).astype(int)
    df["is_paid"] = (~df["utm_medium"].isin(["organic", "referral", "(none)"])).astype(int)

    # Признаки, связанные с номером визита
    df["is_first_visit"] = (df["visit_number"] == 1).astype(int)
    df["is_returning"] = (df["visit_number"] > 3).astype(int)

    # Признаки времени суток
    df["is_night"] = df["visit_hour"].isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
    df["is_work_hours"] = df["visit_hour"].between(10, 19).astype(int)

    # Признаки устройства
    df["is_mobile"] = (df["device_category"] == "mobile").astype(int)
    df["is_desktop"] = (df["device_category"] == "desktop").astype(int)

    # Логарифм номера визита помогает сгладить сильную асимметрию
    df["log_visit_number"] = np.log1p(df["visit_number"])

    return df


def time_split(df: pd.DataFrame, train_size: float = 0.8):
    # Берём только строки, где удалось корректно собрать datetime
    valid_idx = df["visit_datetime"].notna()

    df_valid = df.loc[valid_idx].copy()

    # Сортируем по времени визита
    df_valid = df_valid.sort_values("visit_datetime").reset_index(drop=True)

    # Находим границу между train и test
    split_point = int(len(df_valid) * train_size)

    df_train = df_valid.iloc[:split_point].copy()
    df_test = df_valid.iloc[split_point:].copy()

    return df_train, df_test


def build_group_values(train_df: pd.DataFrame) -> dict:
    group_values = {}

    for col, top_n in GROUP_TOP_N.items():
        series = train_df[col].fillna("unknown").astype(str)
        top_values = series.value_counts().head(top_n).index.tolist()
        group_values[col] = top_values

    return group_values


def apply_grouping(df: pd.DataFrame, group_values: dict) -> pd.DataFrame:
    """
    Применяет укрупнение категорий:
    - частые значения оставляются как есть
    - редкие объединяются в "other"
    """
    df = df.copy()

    for col, values in group_values.items():
        source_series = df[col].fillna("unknown").astype(str)
        grouped_col = f"{col}_group"
        df[grouped_col] = source_series.where(source_series.isin(values), other="other")

    return df


def select_final_features(df: pd.DataFrame):
    """
    Выбирает финальные признаки и target.
    Также делит признаки на числовые и категориальные.
    """
    X = df[FINAL_FEATURES].copy()
    y = df["has_target"].copy()

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Строит общий препроцессор:
    - числовые признаки: заполнение медианой + стандартизация
    - категориальные признаки: заполнение модой + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor, model_name: str) -> dict:
    """
    Обучает одну модель и считает основные метрики качества.
    """
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Обучение
    pipe.fit(X_train, y_train)

    # Предсказания классов и вероятностей
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Pipeline": pipe
    }


def save_feature_importance(best_model_name: str, best_pipeline: Pipeline) -> None:
    """
    Сохраняет интерпретацию признаков для лучшей модели:
    - importance для деревьев
    - коэффициенты для логистической регрессии
    """
    preprocessor_step = best_pipeline.named_steps["preprocessor"]
    model_step = best_pipeline.named_steps["model"]
    feature_names = preprocessor_step.get_feature_names_out()

    if best_model_name in ["RandomForestClassifier", "GradientBoostingClassifier"]:
        values = model_step.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": values
        }).sort_values(by="Importance", ascending=False)

        df_imp.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)
        print("\nТоп-20 важных признаков:")
        print(df_imp.head(20))

    elif best_model_name == "LogisticRegression":
        values = model_step.coef_[0]
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": values,
            "AbsCoefficient": np.abs(values)
        }).sort_values(by="AbsCoefficient", ascending=False)

        df_imp.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)
        print("\nТоп-20 признаков по модулю коэффициента:")
        print(df_imp.head(20))


def train_final_model_on_full_data(df_full: pd.DataFrame, best_model_name: str, group_values_full: dict):
    """
    После выбора лучшей модели обучает её заново на всём доступном наборе данных
    и сохраняет артефакт для API.
    """
    # Применяем группировки
    df_full_grouped = apply_grouping(df_full, group_values_full)

    # Получаем финальные признаки
    X_full, y_full, numeric_features, categorical_features = select_final_features(df_full_grouped)

    # Строим препроцессор
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Повторяем словарь моделей, чтобы взять лучшую
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            max_iter=3000,
            class_weight="balanced"
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=42
        )
    }

    final_model = models[best_model_name]

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", final_model)
    ])

    # Финальное обучение на всём наборе
    final_pipeline.fit(X_full, y_full)

    # Сохраняем не только модель, но и метаданные,
    # чтобы API мог правильно готовить признаки
    artifact = {
        "pipeline": final_pipeline,
        "group_values": group_values_full,
        "feature_columns": FINAL_FEATURES,
        "group_top_n": GROUP_TOP_N,
        "best_model_name": best_model_name,
    }

    joblib.dump(artifact, MODEL_OUTPUT)
    print(f"\nФинальный артефакт сохранён в {MODEL_OUTPUT}")


def main() -> None:
    """
    Основной сценарий обучения:
    1. загрузка данных
    2. построение признаков
    3. time split
    4. группировка категорий
    5. обучение и сравнение моделей
    6. сохранение результатов
    7. обучение финальной модели на полном наборе данных
    """
    print("1. Загружаю данные...")
    df = load_data()
    print("Размер после объединения:", df.shape)

    print("\n2. Создаю безопасные признаки...")
    df = create_base_features(df)

    print("\n3. Делаю честное временное разбиение...")
    df_train, df_test = time_split(df)

    print("Train:", df_train.shape)
    print("Test:", df_test.shape)

    print("\n4. Строю группировки редких категорий по train...")
    group_values_train = build_group_values(df_train)

    df_train = apply_grouping(df_train, group_values_train)
    df_test = apply_grouping(df_test, group_values_train)

    print("\n5. Формирую финальные признаки...")
    X_train, y_train, numeric_features, categorical_features = select_final_features(df_train)
    X_test, y_test, _, _ = select_final_features(df_test)

    print("Числовых признаков:", len(numeric_features))
    print("Категориальных признаков:", len(categorical_features))
    print("Размер X_train:", X_train.shape)
    print("Размер X_test:", X_test.shape)

    print("\nБаланс целевой переменной:")
    print(y_train.value_counts(normalize=True))

    print("\n6. Строю препроцессор...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    print("\n7. Сравниваю модели...")
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            max_iter=3000,
            class_weight="balanced"
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=42
        )
    }

    results = []

    for model_name, model in models.items():
        print(f"Обучаю {model_name} ...")
        result = evaluate_model(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            model_name=model_name
        )
        results.append(result)

    # Формируем итоговую таблицу результатов
    results_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"],
            "ROC_AUC": r["ROC_AUC"]
        }
        for r in results
    ]).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)

    print("\nСравнение моделей:")
    print(results_df)

    # Сохраняем таблицу с результатами
    results_df.to_csv(REPORT_OUTPUT, index=False)
    print(f"\nТаблица результатов сохранена в {REPORT_OUTPUT}")

    # Определяем лучшую модель
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nЛучшая модель по ROC-AUC: {best_model_name}")

    best_pipeline = None
    for r in results:
        if r["Model"] == best_model_name:
            best_pipeline = r["Pipeline"]
            break

    # Сохраняем интерпретацию признаков
    save_feature_importance(best_model_name, best_pipeline)

    print("\n8. Обучаю финальную модель на полном наборе данных...")
    full_valid_df = df[df["visit_datetime"].notna()].copy().sort_values("visit_datetime").reset_index(drop=True)
    group_values_full = build_group_values(full_valid_df)
    train_final_model_on_full_data(full_valid_df, best_model_name, group_values_full)


if __name__ == "__main__":
    main()
