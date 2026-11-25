from ultralytics import YOLO
import cv2
import os
import time

print("=== НАКОПИТЕЛЬНЫЙ СБОР ДАТАСЕТА ===\n")

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Камера не открылась")
    exit(1)

SAVE_DIR = "dataset/owner"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. ОПРЕДЕЛЕНИЕ НОМЕРА ФОТОСЕССИИ

while True:
    session_input = ""
    if session_input == "":
        # Авто-определение: ищем максимальный номер сессии в папке
        existing = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')]
        max_s = 0
        for f in existing:
            if f.startswith('s') and '_' in f:
                try:
                    s_num = int(f.split('_')[0][1:])
                    if s_num > max_s: max_s = s_num
                except:
                    pass
        session_id = max_s + 1
        break
    elif session_input.isdigit() and 1 <= int(session_input) <= 99:
        session_id = int(session_input)
        break
    else:
        print("Неверный номер, eblusha")

session_prefix = f"s{session_id:02d}"
print(f"Текущая фотосессия: {session_prefix}\n")

# Режимы: (ID_режима, "Текст на экране", Количество снимков)
MODES = [
    ("01", "1. FULL ROST, FACE TO CAMERA", 25),
    ("02", "2. FULL ROST, LEFT", 20),
    ("03", "3. FULL ROST, RIGHT", 20),
    ("04", "4. FULL ROST, BACK TO CAMERA", 20),
    ("05", "5. WAIST, FACE TO CAMERA", 20),
    ("06", "6. WAIST, BACK TO CAMERA", 20),
    ("07", "7. SIT DOWN", 20),
    ("08", "8. MOVING!", 25),
]

PREPARE_TIME = 1.0  # Секунд на подготовку позы
SHOOT_INTERVAL = 1.5  # Секунд между снимками

new_count = 0
skipped = 0


def draw_overlay(frame, text_main, text_sub, countdown=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text_main, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(frame, text_sub, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if countdown is not None:
        cv2.putText(frame, f"{countdown:.1f}", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
    return frame


total_shots = sum(m[2] for m in MODES)
print(f"Всего режимов: {len(MODES)} | Планируется снимков: {total_shots}")

for mode_id, mode_desc, num_shots in MODES:
    print(f"\n>>> РЕЖИМ {mode_id}: {mode_desc} ({num_shots} снимков)")

    # Пауза перед режимом (5 сек)
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, clean_frame = cap.read()
        if not ret: continue
        display_frame = clean_frame.copy()
        remaining = 5 - (time.time() - start_time)
        display_frame = draw_overlay(display_frame, mode_desc, "READY???...", countdown=remaining)
        cv2.imshow('AUTO COLLECT', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release();
            cv2.destroyAllWindows();
            exit(0)
        time.sleep(0.03)

    for i in range(num_shots):
        # Обратный отсчёт (3 сек)
        start_time = time.time()
        while time.time() - start_time < PREPARE_TIME:
            ret, clean_frame = cap.read()
            if not ret: break
            display_frame = clean_frame.copy()
            remaining = PREPARE_TIME - (time.time() - start_time)
            display_frame = draw_overlay(display_frame, mode_desc, f"FRAME {i + 1}/{num_shots}", countdown=remaining)
            cv2.imshow('AUTO COLLECT', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release();
                cv2.destroyAllWindows();
                exit(0)
            time.sleep(0.03)

        # Проверка и сохранение ЧИСТОГО кадра
        results = model(clean_frame, conf=0.5, verbose=False)
        if len(results[0].boxes) == 0:
            skipped += 1
            continue

        largest = max(results[0].boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest.xyxy[0])
        if (y2 - y1) < 100:  # Игнорируем слишком мелкие детекции
            skipped += 1
            continue

        # УНИКАЛЬНОЕ ИМЯ: s01_01_0000.jpg
        filename = os.path.join(SAVE_DIR, f"{session_prefix}_{mode_id}_{new_count:04d}.jpg")
        cv2.imwrite(filename, clean_frame)
        new_count += 1
        print(f"    ✓ {filename}")

        # Пауза между снимками (2 сек)
        start_time = time.time()
        while time.time() - start_time < SHOOT_INTERVAL:
            ret, clean_frame = cap.read()
            if not ret: break
            display_frame = clean_frame.copy()
            display_frame = draw_overlay(display_frame, mode_desc, f"SAVED {i + 1}/{num_shots}. change pose.",
                                         countdown=None)
            cv2.imshow('AUTO COLLECT', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release();
                cv2.destroyAllWindows();
                exit(0)
            time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()

# Итоговая статистика
total_in_folder = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])

print("\n" + "=" * 60)
print(f"=== ФОТОСЕССИЯ {session_prefix} ЗАВЕРШЕНА ===")
print(f"Добавлено новых фото: {new_count}")
print(f"Пропущено (нет человека/мелко): {skipped}")
print(f"ВСЕГО фото в датасете: {total_in_folder}")
