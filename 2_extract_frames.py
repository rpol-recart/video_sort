import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# --- НАСТРОЙКИ ---
CONFIG = {
    # 1. Пути
    # Директория, где лежат результаты предыдущего скрипта
    "processing_input_dir": "processing_outputs", 
    # Директория, куда будут сохраняться отобранные кадры
    "frame_output_dir": "extracted_frames",
    
    # 2. Имена файлов
    # Входной файл со списком уникальных видео
    "unique_videos_list_file": "unique_video_representatives.txt",
    # Выходной файл со списком путей ко всем извлеченным кадрам
    "final_framelist_file": "final_dataset_framelist.txt",

    # 3. Параметры извлечения и фильтрации
    # Порог для SSIM. Если > этого значения, кадр считается статичным.
    # 0.98 - хороший компромисс. Уменьшайте, если хотите отбрасывать больше кадров,
    # или увеличивайте, если хотите захватить даже малейшие изменения.
    "ssim_static_threshold": 0.98,
    
    # Как часто проверять кадр на уникальность (в секундах). 0.5 = 2 раза в секунду.
    "extraction_interval_sec": 0.5,
    
    # Размер кадра для быстрого сравнения по SSIM. Не влияет на разрешение сохраняемого кадра.
    "ssim_compare_resolution": (224, 224),

    # Качество сохраняемых JPEG-файлов (0-100)
    "jpeg_quality": 90
}


def extract_and_filter_frames(task_args):
    """
    Основная функция-worker. Принимает путь к видео, извлекает из него кадры,
    фильтрует статичные по SSIM и сохраняет уникальные.
    """
    video_path, output_dir = task_args
    
    saved_frame_paths = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # print(f"Не удалось открыть видео: {video_path}")
            return []

        # Создаем уникальную подпапку для кадров из этого видео, чтобы избежать конфликтов имен
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_sub_dir = os.path.join(output_dir, video_name)
        os.makedirs(frame_sub_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Если FPS=0, видео повреждено или имеет необычный формат
        if fps == 0 or fps > 200: 
            cap.release()
            return []
            
        frame_skip = int(fps * CONFIG["extraction_interval_sec"])
        if frame_skip == 0:
            frame_skip = 1

        last_saved_frame_gray = None
        frame_count = 0
        
        while True:
            # Читаем кадр только если он нам нужен для проверки
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break # Видео закончилось

            # Работаем с уменьшенными Ч/Б изображениями для быстрой оценки SSIM
            resized = cv2.resize(frame, CONFIG["ssim_compare_resolution"], interpolation=cv2.INTER_AREA)
            current_frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            is_unique = False
            # Первый кадр всегда уникален
            if last_saved_frame_gray is None:
                is_unique = True
            else:
                # Сравниваем текущий кадр с ПОСЛЕДНИМ СОХРАНЕННЫМ, а не с предыдущим
                score = ssim(last_saved_frame_gray, current_frame_gray)
                if score < CONFIG["ssim_static_threshold"]:
                    is_unique = True
            
            if is_unique:
                frame_filename = f"frame_{frame_count:06d}.jpg"
                save_path = os.path.join(frame_sub_dir, frame_filename)
                
                # Сохраняем ОРИГИНАЛЬНЫЙ кадр в высоком качестве
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]])
                saved_frame_paths.append(save_path)
                
                # Обновляем "эталон" для сравнения
                last_saved_frame_gray = current_frame_gray

            frame_count += frame_skip
            
        cap.release()
        return saved_frame_paths
    except Exception as e:
        # Подавляем ошибки для отдельных видео, чтобы не останавливать весь процесс
        # print(f"Ошибка при извлечении кадров из {video_path}: {e}")
        return []

def main():
    """Основная логика выполнения скрипта."""
    start_time = time.time()
    
    input_list_path = os.path.join(CONFIG["processing_input_dir"], CONFIG["unique_videos_list_file"])
    
    print(f"--- ЭТАП 3: Извлечение и фильтрация кадров ---")
    print(f"Чтение списка видео из файла: {input_list_path}")

    try:
        with open(input_list_path, 'r') as f:
            # Убираем пустые строки и символы переноса
            videos_to_process = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ОШИБКА: Входной файл не найден: {input_list_path}")
        print("Пожалуйста, сначала запустите скрипт '1_cluster_videos.py'.")
        return

    if not videos_to_process:
        print("Входной файл пуст. Нет видео для обработки.")
        return

    print(f"Найдено {len(videos_to_process)} уникальных видео для обработки.")
    os.makedirs(CONFIG["frame_output_dir"], exist_ok=True)
    
    # Подготовка задач для параллельной обработки
    tasks = [(path, CONFIG["frame_output_dir"]) for path in videos_to_process]
    all_saved_frames = []

    print(f"Запуск извлечения кадров с использованием {cpu_count()} процессов...")
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(tasks), desc="Извлечение кадров") as pbar:
            # imap_unordered быстрее, т.к. обрабатывает результаты по мере их поступления
            for frame_list_result in pool.imap_unordered(extract_and_filter_frames, tasks):
                all_saved_frames.extend(frame_list_result)
                pbar.set_postfix(extracted=f"{len(all_saved_frames):,}")
                pbar.update()

    # --- Итоги и сохранение результата ---
    final_list_path = os.path.join(CONFIG["processing_input_dir"], CONFIG["final_framelist_file"])
    print("\nСохранение финального списка путей к кадрам...")
    with open(final_list_path, 'w') as f:
        for path in all_saved_frames:
            f.write(f"{path}\n")

    end_time = time.time()
    print("\n--- ЗАВЕРШЕНО ---")
    print(f"Обработано видео: {len(videos_to_process)}")
    print(f"Итого отобрано и сохранено уникальных кадров: {len(all_saved_frames)}")
    
    duration_secs = end_time - start_time
    m, s = divmod(duration_secs, 60)
    h, m = divmod(m, 60)
    print(f"Общее время выполнения: {int(h)} ч, {int(m)} мин, {int(s)} сек.")
    print(f"Кадры сохранены в директории: {CONFIG['frame_output_dir']}")
    print(f"Финальный список кадров для DINOv2 сохранен в: '{final_list_path}'")

if __name__ == "__main__":
    main()