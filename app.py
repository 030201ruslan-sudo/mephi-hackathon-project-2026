# Path нужен для корректной работы с путями к файлам
from pathlib import Path

# traceback поможет выводить подробную ошибку в JSON-ответе API
import traceback

# Библиотеки для загрузки модели, подготовки признаков и запуска API
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Папка проекта
BASE_DIR = Path(__file__).resolve().parent

# Файл с сохранённым артефактом модели
MODEL_PATH = BASE_DIR / "best_conversion_model.joblib"

# Создаём Flask-приложение
app = Flask(__name__)

# Глобальные переменные, которые будут заполнены после загрузки артефакта
artifact = None
model = None
group_values = None
feature_columns = None


def load_model():

    global artifact, model, group_values, feature_columns

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict) and "pipeline" in artifact:
        model = artifact["pipeline"]
        group_values = artifact["group_values"]
        feature_columns = artifact["feature_columns"]
    else:
        raise ValueError(
            "Ожидался новый формат артефакта. "
            "Сначала заново запустите train_model.py."
        )


def group_with_known_values(value, allowed_values):

    value = "unknown" if value is None else str(value)
    return value if value in allowed_values else "other"


def build_features_from_json(payload: dict) -> pd.DataFrame:

    row = payload.copy()

    # Дата и время визита приходят из JSON
    visit_date = row.get("visit_date")
    visit_time = row.get("visit_time")

    # Собираем visit_datetime
    visit_datetime = pd.to_datetime(
        f"{visit_date} {visit_time}",
        errors="coerce"
    )

    # Если дата/время некорректны, возвращаем ошибку
    if pd.isna(visit_datetime):
        raise ValueError(
            "Не удалось распознать visit_date и visit_time. "
            "Ожидаемый формат: visit_date='2024-01-31', visit_time='14:35:00'"
        )

    # Номер визита по умолчанию = 1
    visit_number = row.get("visit_number", 1)
    try:
        visit_number = int(visit_number)
    except Exception:
        visit_number = 1

    utm_medium = row.get("utm_medium", "unknown")
    device_category = row.get("device_category", "unknown")

    # Собираем все признаки в том виде, как они использовались при обучении
    prepared = {
        "visit_number": visit_number,
        "log_visit_number": float(np.log1p(visit_number)),
        "visit_hour": int(visit_datetime.hour),
        "visit_dayofweek": int(visit_datetime.dayofweek),
        "visit_month": int(visit_datetime.month),
        "is_weekend": int(visit_datetime.dayofweek in [5, 6]),
        "is_organic": int(utm_medium in ["organic", "referral", "(none)"]),
        "is_paid": int(utm_medium not in ["organic", "referral", "(none)"]),
        "is_first_visit": int(visit_number == 1),
        "is_returning": int(visit_number > 3),
        "is_night": int(visit_datetime.hour in [0, 1, 2, 3, 4, 5, 6]),
        "is_work_hours": int(10 <= visit_datetime.hour <= 19),
        "is_mobile": int(device_category == "mobile"),
        "is_desktop": int(device_category == "desktop"),
        "device_category": str(device_category),
        "device_os": str(row.get("device_os", "unknown")),

        # Ниже идут укрупнённые категориальные признаки.
        # Здесь мы используем те же group_values, что были получены при обучении.
        "device_browser_group": group_with_known_values(
            row.get("device_browser"), group_values["device_browser"]
        ),
        "geo_country_group": group_with_known_values(
            row.get("geo_country"), group_values["geo_country"]
        ),
        "geo_city_group": group_with_known_values(
            row.get("geo_city"), group_values["geo_city"]
        ),
        "utm_source_group": group_with_known_values(
            row.get("utm_source"), group_values["utm_source"]
        ),
        "utm_medium_group": group_with_known_values(
            row.get("utm_medium"), group_values["utm_medium"]
        ),
        "utm_campaign_group": group_with_known_values(
            row.get("utm_campaign"), group_values["utm_campaign"]
        ),
    }

    # Приводим всё к DataFrame с тем же порядком колонок, что и при обучении
    df = pd.DataFrame([prepared], columns=feature_columns)
    return df


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "message": "Conversion prediction API is running",
        "endpoints": {
            "GET /health": "Проверка работоспособности",
            "POST /predict": "Предсказание класса и вероятности"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        if model is None:
            return jsonify({
                "error": "Модель не загружена"
            }), 500

        # Читаем JSON из запроса
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({
                "error": "Ожидается JSON в теле запроса"
            }), 400

        # Готовим признаки
        features_df = build_features_from_json(payload)

        # Получаем предсказание класса и вероятность
        predicted_class = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])

        return jsonify({
            "predicted_class": predicted_class,
            "conversion_probability": probability
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# Пытаемся загрузить модель сразу при старте приложения
try:
    load_model()
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")


if __name__ == "__main__":
    # Локальный запуск API
    app.run(host="0.0.0.0", port=5000, debug=True)