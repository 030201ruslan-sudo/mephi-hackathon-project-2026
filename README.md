Sber Autosubscription Hackathon
Описание проекта

В проекте решается задача прогнозирования вероятности совершения пользователем целевого действия на сайте «СберАвтоподписка».

Признаки из ga_hits.csv, агрегированные по всей сессии, не используются в модели, чтобы избежать утечки данных.
Файл ga_hits.csv используется только для построения целевой переменной has_target.

Состав проекта
01_eda_analysis.ipynb - разведочный анализ данных
02_modeling.ipynb - построение и сравнение моделей
prepare_target.py - формирование целевой переменной из ga_hits.csv
train_model.py - обучение и сохранение финальной модели
app.py - REST API для предсказания
best_conversion_model.joblib - сохранённый артефакт модели
model_comparison.csv - таблица результатов моделей
feature_importance.csv - интерпретация признаков лучшей модели

Установка зависимостей

python3 -m pip install pandas numpy scikit-learn joblib flask matplotlib seaborn openpyxl

Порядок запуска

1. Подготовить целевую переменную
python3 prepare_target.py

Будет создан файл:

session_target.csv

2. Обучить модель
python3 train_model.py

Будут созданы файлы:

best_conversion_model.joblib
model_comparison.csv
feature_importance.csv

3. Запустить API
python3 app.py

После запуска API будет доступен по адресу:

http://127.0.0.1:5000
Доступные endpoints
GET /

Возвращает общую информацию о сервисе.

GET /health

Проверка работоспособности API.

POST /predict

Предсказание вероятности совершения целевого действия.

Пример запроса
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "visit_date": "2021-05-24",
  "visit_time": "14:35:00",
  "visit_number": 1,
  "utm_source": "google",
  "utm_medium": "cpc",
  "utm_campaign": "summer_sale",
  "device_category": "desktop",
  "device_os": "Windows",
  "device_browser": "Chrome",
  "geo_country": "Russia",
  "geo_city": "Moscow"
}'
Пример ответа
{
  "predicted_class": 0,
  "conversion_probability": 0.1374
}
Подход к моделированию

В модели используются только безопасные признаки, доступные на момент визита пользователя:

номер визита;
признаки времени визита;
характеристики источника трафика;
признаки устройства;
география;
инженерные бинарные признаки;
укрупнённые категориальные признаки (*_group).

Разбиение на обучающую и тестовую выборки выполняется по времени, чтобы оценка качества была ближе к реальному сценарию прогнозирования будущих визитов.

Результат

Лучшая модель выбирается по метрике ROC-AUC.
После устранения утечки данных качество модели стало более реалистичным и отражает её реальную предсказательную способность.