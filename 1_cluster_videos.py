import os
import cv2
import json
import imagehash
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# --- НАСТРОЙКИ ---
CONFIG = {
    # 1. Пути
    "video_source_dir": "/data/raw_videos",
    "output_dir": "/data/processing_outputs",

    # 2. Параметры создания отпечатков
    # На каких секундах видео брать кадры для "отпечатка"
    "fingerprint_points_sec": [5, 15, 25, 35, 45, 55],
    
    # 3. Параметры кластеризации
    # Порог. Если среднее расстояние Хэмминга < этого значения, видео считаются дубликатами.
    # Хорошие значения для старта: 3, 4, 5.
    "phash_cluster_threshold": 4,

    # 4. Имена выходных файлов
    # Кэш всех отпечатков, чтобы не пересчитывать их каждый раз
    "fingerprints_cache_file": "video_fingerprints.json",
    # Итоговый список уникальных видео для следующего этапа
    "unique_videos_list_file": "unique_video_representatives.txt"
}

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_video_fingerprint(video_path):
    """
    Извлекает кадры из видео, вычисляет их pHash и возвращает как "отпечаток".
    Возвращает кортеж (путь_к_видео, список_хешей_в_виде_строк).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, [])

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Если FPS не читается, видео скорее всего повреждено
        if fps == 0:
            cap.release()
            return (video_path, [])
            
        fingerprint_hashes = []
        for sec in CONFIG["fingerprint_points_sec"]:
            frame_id = int(fps * sec)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                # Конвертация BGR (OpenCV) в RGB (Pillow) и вычисление хеша
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Конвертируем хеш в строку для совместимости с JSON
                fingerprint_hashes.append(str(imagehash.phash(pil_img)))
        
        cap.release()
        return (video_path, fingerprint_hashes)
    except Exception:
        # Подавляем ошибки для поврежденных файлов, просто возвращая пустой результат
        return (video_path, [])

def compare_fingerprints(fp1_str, fp2_str):
    """
    Вычисляет среднее расстояние Хэмминга между двумя отпечатками (списками хеш-строк).
    """
    if len(fp1_str) != len(fp2_str) or not fp1_str:
        return float('inf')
    
    # Конвертируем строки обратно в объекты ImageHash для сравнения
    fp1_hashes = [imagehash.hex_to_hash(h) for h in fp1_str]
    fp2_hashes = [imagehash.hex_to_hash(h) for h in fp2_str]
    
    # Библиотека imagehash позволяет вычитать хеши для получения расстояния Хэмминга
    distances = [h1 - h2 for h1, h2 in zip(fp1_hashes, fp2_hashes)]
    return sum(distances) / len(distances)

# --- ОСНОВНОЙ СКРИПТ ---
def main():
    """Основная логика выполнения скрипта."""
    start_time = time.time()
    
    # Создаем выходную директорию, если ее нет
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    cache_path = os.path.join(CONFIG["output_dir"], CONFIG["fingerprints_cache_file"])

    # --- ЭТАП 1: Создание или загрузка отпечатков для всех видео ---
    print("--- ЭТАП 1: Создание/Загрузка отпечатков видео ---")
    
    video_fingerprints = []
    
    if os.path.exists(cache_path):
        print(f"Найден файл с кэшем отпечатков. Загрузка из '{cache_path}'...")
        with open(cache_path, 'r') as f:
            video_fingerprints = json.load(f)
        print(f"Загружено {len(video_fingerprints)} отпечатков.")
    else:
        print("Файл с кэшем не найден. Начинается сканирование видео...")
        video_files = [
            os.path.join(CONFIG["video_source_dir"], f) 
            for f in os.listdir(CONFIG["video_source_dir"]) 
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        
        if not video_files:
            print(f"В директории '{CONFIG['video_source_dir']}' не найдено видео. Завершение работы.")
            return

        print(f"Найдено {len(video_files)} видео для обработки.")
        
        # Запускаем обработку в несколько процессов
        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=len(video_files), desc="Создание отпечатков") as pbar:
                for path, fp in pool.imap_unordered(get_video_fingerprint, video_files):
                    # Добавляем только успешно обработанные видео с полным отпечатком
                    if fp and len(fp) == len(CONFIG["fingerprint_points_sec"]):
                        video_fingerprints.append({"path": path, "fp": fp})
                    pbar.update()
        
        print(f"Сохранение {len(video_fingerprints)} отпечатков в кэш-файл '{cache_path}'...")
        with open(cache_path, 'w') as f:
            json.dump(video_fingerprints, f, indent=2)

    print(f"Этап 1 завершен. Общее количество валидных видео: {len(video_fingerprints)}")

    # --- ЭТАП 2: Кластеризация и отбор уникальных видео ---
    print("\n--- ЭТАП 2: Кластеризация и отбор уникальных видео ---")
    if not video_fingerprints:
        print("Нет данных для кластеризации. Завершение работы.")
        return

    clusters = []
    for video_data in tqdm(video_fingerprints, desc="Кластеризация видео"):
        found_cluster = False
        for cluster in clusters:
            # Сравниваем новое видео с представителем (первым элементом) каждого кластера
            representative_fp = cluster[0]["fp"]
            dist = compare_fingerprints(video_data["fp"], representative_fp)
            
            if dist < CONFIG["phash_cluster_threshold"]:
                cluster.append(video_data)
                found_cluster = True
                break
        
        if not found_cluster:
            # Если похожего кластера не найдено, создаем новый
            clusters.append([video_data])
            
    # Собираем пути к представителям кластеров
    unique_video_representatives = [cluster[0]["path"] for cluster in clusters]
    
    print(f"Кластеризация завершена. Найдено {len(unique_video_representatives)} уникальных кластеров.")

    # --- Сохранение результатов ---
    output_list_path = os.path.join(CONFIG["output_dir"], CONFIG["unique_videos_list_file"])
    print(f"Сохранение списка представителей кластеров в файл '{output_list_path}'...")
    
    with open(output_list_path, 'w') as f:
        for path in unique_video_representatives:
            f.write(f"{path}\n")

    end_time = time.time()
    print("\n--- ЗАВЕРШЕНО ---")
    print(f"Общее время выполнения:{round(end_time - start_time, 2)} секунд.")
    # Более читаемый формат времени
    duration_secs = end_time - start_time
    m, s = divmod(duration_secs, 60)
    h, m = divmod(m, 60)
    print(f"Общее время выполнения: {int(h)} ч, {int(m)} мин, {int(s)} сек.")
    print(f"Результаты сохранены в директории: {CONFIG['output_dir']}")
    print(f"Файл со списком уникальных видео для Этапа 3: '{output_list_path}'")

if __name__ == "__main__":
    main()