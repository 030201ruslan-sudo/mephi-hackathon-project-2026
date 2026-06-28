# Библиотека pandas нужна для чтения CSV и работы с таблицами
import pandas as pd

# Список целевых действий, которые считаются конверсией
TARGET_ACTIONS = {
    "sub_car_claim_click",
    "sub_car_claim_submit_click",
    "sub_open_dialog_click",
    "sub_custom_question_submit_click",
    "sub_call_number_click",
    "sub_callback_submit_click",
    "sub_submit_success",
    "sub_car_request_submit_click",
}

# Входной и выходной файлы
INPUT_FILE = "ga_hits.csv"
OUTPUT_FILE = "session_target.csv"

# Читаем ga_hits.csv частями, чтобы не перегружать память
CHUNKSIZE = 200_000


def main() -> None:
    """
    Строит целевую переменную has_target на уровне session_id.

    Логика:
    - если в сессии было хотя бы одно действие из TARGET_ACTIONS,
      то has_target = 1
    - иначе has_target = 0
    """
    print("Начинаю обработку ga_hits.csv ...")

    # Здесь будем хранить результат по каждой сессии:
    # session_id -> 0 или 1
    session_flags = {}

    # Читаем файл по частям
    for i, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=["session_id", "event_action"],
            chunksize=CHUNKSIZE,
            low_memory=False
        ),
        start=1
    ):
        # Проверяем, относится ли событие к целевым
        chunk["has_target_action"] = chunk["event_action"].isin(TARGET_ACTIONS).astype(int)

        # Для каждой session_id оставляем максимум:
        # если хотя бы одно целевое действие было, то максимум будет 1
        grouped = chunk.groupby("session_id")["has_target_action"].max()

        # Обновляем общий словарь по сессиям
        for session_id, flag in grouped.items():
            if session_id in session_flags:
                session_flags[session_id] = max(session_flags[session_id], int(flag))
            else:
                session_flags[session_id] = int(flag)

        print(f"Обработан chunk #{i}, текущих session_id: {len(session_flags)}")

    # Переводим словарь в DataFrame
    target_df = pd.DataFrame({
        "session_id": list(session_flags.keys()),
        "has_target": list(session_flags.values())
    })

    # Сохраняем результат
    target_df.to_csv(OUTPUT_FILE, index=False)

    print("\nГотово.")
    print("Размер итоговой таблицы:", target_df.shape)
    print("Распределение target:")
    print(target_df["has_target"].value_counts())
    print(f"\nФайл сохранён: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()