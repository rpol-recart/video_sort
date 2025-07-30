import os
import cv2
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
from multiprocessing import Process, Queue, Manager
from collections import defaultdict
import shutil

# --- НАСТРОЙКИ ---
CONFIG = {
    # 1. Список RTSP-потоков с камер
    # 'cam_01', 'cam_02' - это будут имена папок для каждой камеры
    "rtsp_streams": {
        "cam_01": "rtsp://user:password@192.168.1.10:554/stream1",
        "cam_02": "rtsp://user:password@192.168.1.11:554/stream1",
        # Для теста можно использовать локальный файл, OpenCV это позволяет
        # "cam_test": "/path/to/your/test/video.mp4" 
    },

    # 2. Пути
    "output_base_dir": "/data/camera_captures", # Главная папка для всех кадров

    # 3. Параметры фильтрации и кластеризации
    # --- Фильтр Времени (для отсеивания статики) ---
    "ssim_threshold": 0.96, # Чем НИЖЕ, тем более значительные изменения нужны, чтобы сохранить кадр.
    "ssim_compare_resolution": (224, 224), # Разрешение для быстрого сравнения SSIM

    # --- Фильтр Сцены (для группировки по кластерам) ---
    "phash_threshold": 6, # Если расстояние Хэмминга < этого значения, кадры считаются одной сценой.

    # 4. Параметры работы
    "reconnect_delay_sec": 10, # Пауза перед попыткой переподключения к камере
    "jpeg_quality": 85,
}

# --- КОМПОНЕНТ 1: ВОРКЕР ДЛЯ КАЖДОЙ КАМЕРЫ (ФИЛЬТР ВРЕМЕНИ) ---

def camera_stream_worker(camera_id, rtsp_url, frame_queue):
    """
    Этот процесс подключается к одной камере, фильтрует статику и передает
    уникальные кадры в общую очередь.
    """
    print(f"[{camera_id}] Запуск воркера для {rtsp_url}")
    
    # Создаем папку для временного хранения кадров с этой камеры
    temp_output_dir = os.path.join(CONFIG["output_base_dir"], "_temp", camera_id)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    last_saved_frame_gray = None
    
    while True: # Бесконечный цикл для переподключения
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[{camera_id}] Ошибка: не удалось подключиться к камере. Повтор через {CONFIG['reconnect_delay_sec']} сек...")
            time.sleep(CONFIG['reconnect_delay_sec'])
            continue

        print(f"[{camera_id}] Успешно подключено к камере.")
        
        while True: # Основной цикл чтения кадров
            ret, frame = cap.read()
            if not ret:
                print(f"[{camera_id}] Потерян сигнал с камеры. Переподключение...")
                break # Выход из внутреннего цикла для переподключения

            # Уменьшаем кадр до Ч/Б для быстрого сравнения
            resized = cv2.resize(frame, CONFIG["ssim_compare_resolution"], interpolation=cv2.INTER_AREA)
            current_frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            is_unique = False
            # Первый кадр после подключения всегда уникален
            if last_saved_frame_gray is None:
                is_unique = True
            else:
                score = ssim(last_saved_frame_gray, current_frame_gray)
                if score < CONFIG["ssim_threshold"]:
                    is_unique = True
            
            if is_unique:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                frame_filename = f"{camera_id}_{timestamp}.jpg"
                save_path = os.path.join(temp_output_dir, frame_filename)
                
                # Сохраняем оригинальный кадр
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]])
                
                # Отправляем путь в очередь для организатора
                frame_queue.put(save_path)
                
                # Обновляем эталон для сравнения
                last_saved_frame_gray = current_frame_gray
            
            # Небольшая пауза, чтобы не перегружать CPU. Можно убрать, если процессор мощный.
            time.sleep(0.1) 

        cap.release()
        last_saved_frame_gray = None # Сбрасываем эталон при переподключении

# --- КОМПОНЕНТ 2: ОРГАНИЗАТОР (ФИЛЬТР СЦЕНЫ) ---

def frame_organizer(frame_queue, cluster_db):
    """
    Этот процесс забирает кадры из очереди и раскладывает их по папкам-кластерам.
    """
    print("[ORGANIZER] Запуск организатора кадров.")
    
    while True:
        try:
            # Блокируется, пока в очереди не появится новый элемент
            frame_path = frame_queue.get()
            
            if not os.path.exists(frame_path):
                continue
                
            camera_id = os.path.basename(os.path.dirname(frame_path))
            
            # Вычисляем pHash для нового кадра
            try:
                img = Image.open(frame_path)
                new_phash = imagehash.phash(img)
            except Exception as e:
                print(f"[ORGANIZER] Ошибка при обработке кадра {frame_path}: {e}")
                os.remove(frame_path) # Удаляем битый кадр
                continue

            found_cluster = False
            min_dist = float('inf')
            best_cluster_id = None

            # Ищем наиболее похожий существующий кластер
            for cluster_id, representative_phash_str in cluster_db.get(camera_id, {}).items():
                representative_phash = imagehash.hex_to_hash(representative_phash_str)
                dist = new_phash - representative_phash
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_id = cluster_id
            
            # Если найден достаточно похожий кластер
            if best_cluster_id is not None and min_dist < CONFIG["phash_threshold"]:
                cluster_dir = os.path.join(CONFIG["output_base_dir"], camera_id, best_cluster_id)
                found_cluster = True
            else:
                # Создаем новый кластер
                new_cluster_id = f"cluster_{len(cluster_db.get(camera_id, {})):04d}"
                cluster_dir = os.path.join(CONFIG["output_base_dir"], camera_id, new_cluster_id)
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Обновляем базу данных кластеров для этой камеры
                # (в реальном приложении это была бы база данных, а не словарь в памяти)
                if camera_id not in cluster_db:
                    cluster_db[camera_id] = {}
                camera_clusters = cluster_db[camera_id]
                camera_clusters[new_cluster_id] = str(new_phash)
                cluster_db[camera_id] = camera_clusters # Важно для синхронизации Manager dict
                
                print(f"[ORGANIZER] Новая сцена для {camera_id}! Создан кластер {new_cluster_id}")

            # Перемещаем файл из временной папки в папку кластера
            shutil.move(frame_path, os.path.join(cluster_dir, os.path.basename(frame_path)))

        except Exception as e:
            print(f"[ORGANIZER] Критическая ошибка в цикле организатора: {e}")
            time.sleep(5)


# --- КОМПОНЕНТ 3: ТОЧКА ВХОДА И ОРКЕСТРАТОР ПРОЦЕССОВ ---

if __name__ == "__main__":
    # Используем Manager для создания разделяемых между процессами очереди и словаря
    with Manager() as manager:
        frame_queue = manager.Queue()
        # Этот словарь будет хранить pHash'и представителей каждого кластера для каждой камеры
        cluster_db = manager.dict() 

        processes = []

        # 1. Запускаем один процесс-организатор
        organizer_process = Process(target=frame_organizer, args=(frame_queue, cluster_db))
        organizer_process.start()
        processes.append(organizer_process)

        # 2. Запускаем по одному воркеру на каждую камеру
        for camera_id, rtsp_url in CONFIG["rtsp_streams"].items():
            worker_process = Process(target=camera_stream_worker, args=(camera_id, rtsp_url, frame_queue))
            worker_process.start()
            processes.append(worker_process)
            
        print(f"Запущено {len(CONFIG['rtsp_streams'])} воркеров и 1 организатор. Нажмите Ctrl+C для остановки.")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nПолучен сигнал остановки. Завершение работы...")
            for p in processes:
                p.terminate() # Принудительно завершаем все процессы
            print("Все процессы остановлены.")