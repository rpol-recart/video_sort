import os
import cv2
import json
import imagehash
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import argparse  # Добавлен для аргументов командной строки

# --- НАСТРОЙКИ ---
CONFIG = {
    # 1. Пути
    "video_source_dir": "video",
    "output_dir": "processing_outputs",

    # 2. Параметры создания отпечатков
    # На каких секундах видео брать кадры для "отпечатка"
    "fingerprint_points_sec": [5, 15, 25, 35, 45, 55],
    "fingerprint_points_count" : 100,
    
    # 3. Параметры кластеризации
    # Порог. Если среднее расстояние Хэмминга < этого значения, видео считаются дубликатами.
    # Хорошие значения для старта: 3, 4, 5.
    "phash_cluster_threshold": 4,

    # 4. Имена выходных файлов
    # Кэш всех отпечатков в формате JSON Lines для возможности дозаписи
    "fingerprints_cache_file": "video_fingerprints.jsonl",
    # Итоговый список уникальных видео для следующего этапа
    "unique_videos_list_file": "unique_video_representatives.txt"
}

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_video_fingerprint(video_path):
    """
    Извлекает заданное количество кадров, равномерно распределенных по длине видео,
    вычисляет их pHash и возвращает "отпечаток".
    Подходит для видео любой длины.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # print(f"Не удалось открыть видео: {video_path}")
            return (video_path, [])

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Проверка на жизнеспособность видеофайла
        if fps == 0 or frame_count < CONFIG["fingerprint_points_count"]:
            cap.release()
            # print(f"Видео слишком короткое или повреждено: {video_path}")
            return (video_path, [])
            
        fingerprint_hashes = []
        points_count = CONFIG["fingerprint_points_count"]
        
        # Генерируем точки для взятия кадров
        # Мы берем кадры из промежутка от 10% до 95% длительности,
        # чтобы избежать вступительных и заключительных титров.
        for i in range(points_count):
            # Рассчитываем относительную позицию кадра
            # Например, для 6 точек это будут позиции ~0.1, 0.27, 0.44, 0.61, 0.78, 0.95
            relative_position = 0.1 + (i / (points_count - 1)) * 0.85 if points_count > 1 else 0.5
            frame_id = int(frame_count * relative_position)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fingerprint_hashes.append(str(imagehash.phash(pil_img)))
            else:
                # Если какой-то кадр не удалось прочитать, лучше считать отпечаток невалидным
                cap.release()
                return (video_path, [])
        
        cap.release()
        
        # Проверяем, что удалось собрать все запланированные хеши
        if len(fingerprint_hashes) == points_count:
            return (video_path, fingerprint_hashes)
        else:
            return (video_path, [])
            
    except Exception as e:
        # print(f"Ошибка при обработке {video_path}: {e}")
        return (video_path, [])

def compare_fingerprints(fp1_str, fp2_str):
    """
    Вычисляет среднее расстояние Хэмминга между двумя отпечатками (списками хеш-строк).
    """
    if len(fp1_str) != len(fp2_str) or not fp1_str:
        return float('inf')
    
    fp1_hashes = [imagehash.hex_to_hash(h) for h in fp1_str]
    fp2_hashes = [imagehash.hex_to_hash(h) for h in fp2_str]
    
    distances = [h1 - h2 for h1, h2 in zip(fp1_hashes, fp2_hashes)]
    return sum(distances) / len(distances)

# --- ОСНОВНОЙ СКРИПТ ---
def main(args):
    """Основная логика выполнения скрипта."""
    start_time = time.time()
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    cache_path = os.path.join(CONFIG["output_dir"], CONFIG["fingerprints_cache_file"])

    # --- ЭТАП 1: Создание или загрузка отпечатков для всех видео ---
    print("--- ЭТАП 1: Создание/Загрузка отпечатков видео ---")
    
    # Сначала найдем все видеофайлы
    all_video_files = [
        os.path.join(CONFIG["video_source_dir"], f) 
        for f in os.listdir(CONFIG["video_source_dir"]) 
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
    
    if not all_video_files:
        print(f"В директории '{CONFIG['video_source_dir']}' не найдено видео. Завершение работы.")
        return

    print(f"Всего найдено видеофайлов: {len(all_video_files)}")

    # Загружаем уже обработанные отпечатки из кэша
    video_fingerprints = []
    processed_paths = set()
    if os.path.exists(cache_path):
        print(f"Найден файл с кэшем отпечатков. Загрузка из '{cache_path}'...")
        with open(cache_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    video_fingerprints.append(data)
                    processed_paths.add(data['path'])
                except json.JSONDecodeError:
                    print(f"Предупреждение: пропущена поврежденная строка в кэш-файле: {line.strip()}")
        print(f"Загружено {len(video_fingerprints)} уже обработанных отпечатков.")
    
    # Определяем, какие файлы нужно обработать
    files_to_process = [path for path in all_video_files if path not in processed_paths]

    if not files_to_process:
        print("Все видео уже обработаны. Переход к кластеризации.")
    else:
        print(f"Осталось обработать новых видео: {len(files_to_process)}")
        
        # Определяем количество потоков
        num_workers = min(args.workers, cpu_count())
        print(f"Запуск обработки в {num_workers} потоков...")
        
        # Запускаем обработку в несколько процессов
        # Используем режим 'a' (append) для дозаписи в файл
        with open(cache_path, 'a') as cache_file, Pool(processes=num_workers) as pool:
            with tqdm(total=len(files_to_process), desc="Создание отпечатков") as pbar:
                for path, fp in pool.imap_unordered(get_video_fingerprint, files_to_process):
                    # Добавляем только успешно обработанные видео с полным отпечатком
                    if fp:
                        result = {"path": path, "fp": fp}
                        video_fingerprints.append(result)
                        # Немедленно записываем результат в кэш-файл
                        cache_file.write(json.dumps(result) + '\n')
                        cache_file.flush() # Сбрасываем буфер на диск
                    pbar.update()

    print(f"Этап 1 завершен. Общее количество валидных видео: {len(video_fingerprints)}")

    # --- ЭТАП 2: Кластеризация и отбор уникальных видео ---
    print("\n--- ЭТАП 2: Кластеризация и отбор уникальных видео ---")
    if not video_fingerprints:
        print("Нет данных для кластеризации. Завершение работы.")
        return

    clusters = []
    for video_data in tqdm(video_fingerprints, desc="Кластеризация видео"):
        # Пропускаем видео с пустыми отпечатками, если такие вдруг попали
        if not video_data.get("fp"):
            continue

        min_distance = float('inf')
        best_cluster = None

        # Этап 1: Найти лучший кластер (с минимальным расстоянием)
        for cluster in clusters:
            representative_fp = cluster[0]["fp"]
            dist = compare_fingerprints(video_data["fp"], representative_fp)
            
            if dist < min_distance:
                min_distance = dist
                best_cluster = cluster
            
        # Этап 2: Принять решение на основе лучшего найденного варианта
        # Если лучший найденный кластер существует и расстояние до него меньше порога...
        if best_cluster and min_distance < CONFIG["phash_cluster_threshold"]:
            # ...добавляем видео в этот самый лучший кластер.
            best_cluster.append(video_data)
        else:
            # Иначе (если кластеров еще нет или ни один не подошел) - создаем новый.
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
    duration_secs = end_time - start_time
    m, s = divmod(duration_secs, 60)
    h, m = divmod(m, 60)
    print(f"Общее время выполнения: {int(h)} ч, {int(m)} мин, {round(s, 2)} сек.")
    print(f"Результаты сохранены в директории: {CONFIG['output_dir']}")
    print(f"Файл со списком уникальных видео: '{output_list_path}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск дубликатов видео с помощью перцептивных хешей.")
    parser.add_argument('-w', '--workers',
                                    type=int,
                                    default=5, # Значение по умолчанию
                                    help='Количество параллельных процессов для обработки видео. По умолчанию: 5.'
                                    )
    parsed_args = parser.parse_args()
    main(parsed_args)