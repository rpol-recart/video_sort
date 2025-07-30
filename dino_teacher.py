import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import copy
import time

# --- 1. Вспомогательные классы и функции ---

class MultiCropTransform:
    """
    Создает несколько аугментированных "видов" (кропов) одного изображения,
    как это описано в DINO.
    - 2 глобальных кропа (с большим разрешением)
    - Несколько локальных кропов (с низким разрешением)
    """
    def __init__(self,
                 global_crops_scale=(0.4, 1.0),
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8):
        
        # Трансформации для глобальных кропов
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Трансформации для локальных кропов
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

class DINOLoss(nn.Module):
    """
    Реализация функции потерь DINO.
    Студент должен предсказать распределение вероятностей, которое выдает Наставник.
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        # "Голова" наставника не обучается, поэтому создаем буфер, а не параметр
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # student_output: [B_global + B_local, D]
        # teacher_output: [B_global, D]
        
        # Шаг 1: Применяем softmax с температурой
        student_sm = F.log_softmax(student_output / self.student_temp, dim=-1)
        
        # Отсоединяем наставника от графа вычислений и центрируем его выход
        teacher_out_detached = teacher_output.detach()
        teacher_sm = F.softmax((teacher_out_detached - self.center) / self.teacher_temp, dim=-1)
        
        # Шаг 2: Расчет кросс-энтропии
        # Каждый кроп студента должен предсказать выход наставника (для глобальных кропов)
        total_loss = 0
        n_loss_terms = 0
        
        # Сравниваем каждый глобальный кроп студента с каждым глобальным кропом наставника
        for iq, q in enumerate(teacher_sm):
            for v in range(student_output.shape[0]):
                # Пропускаем сравнение одного и того же глобального кропа
                if v == iq:
                    continue
                loss = torch.sum(-q * student_sm[v], dim=-1)
                total_loss += loss
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Шаг 3: Обновляем центр
        self.update_center(teacher_out_detached)
        
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Обновление центра для стабилизации обучения."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def update_teacher_ema(student, teacher, momentum):
    """
    Экспоненциальное скользящее среднее (EMA) для обновления весов Наставника.
    teacher_weights = momentum * teacher_weights + (1 - momentum) * student_weights
    """
    for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
        param_teacher.data.mul_(momentum).add_(param_student.data, alpha=1 - momentum)

# --- 2. Фиктивный датасет (замените на свой) ---

class DummyDataset(Dataset):
    """
    Этот датасет генерирует случайные изображения.
    ЗАМЕНИТЕ ЭТОТ КЛАСС на ваш собственный, который будет читать файлы
    из папки с данными с камер, например, используя `ImageFolder` или собственный код.
    """
    def __init__(self, transform, length=1000):
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Генерируем случайное изображение
        random_image = transforms.ToPILImage()(torch.rand(3, 256, 256))
        return self.transform(random_image)

# --- 3. Основной скрипт обучения ---

if __name__ == '__main__':
    # --- Гиперпараметры ---
    BATCH_SIZE = 4 # Увеличьте, насколько позволяет VRAM (32, 64, 128...)
    EPOCHS = 10
    LEARNING_RATE = 1e-5 # Очень важно использовать низкий learning rate для дообучения!
    EMA_MOMENTUM_START = 0.996
    EMA_MOMENTUM_END = 1.0
    WEIGHT_DECAY = 0.04
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # --- Инициализация моделей ---
    print("Загрузка моделей DINOv2...")
    # Загружаем модель-студента
    student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    
    # Создаем модель-наставника как глубокую копию и загружаем те же веса
    teacher = copy.deepcopy(student)
    
    # Перемещаем модели на GPU
    student.to(device)
    teacher.to(device)
    
    # "Замораживаем" наставника - он не будет обучаться через backpropagation
    for p in teacher.parameters():
        p.requires_grad = False
    
    # --- Подготовка данных ---
    # Создаем наш набор трансформаций Multi-Crop
    transform = MultiCropTransform()
    # Создаем датасет (ЗАМЕНИТЬ НА РЕАЛЬНЫЙ)
    dataset = DummyDataset(transform=transform, length=BATCH_SIZE * 10) # 10 шагов на эпоху
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Инициализация Loss и Оптимизатора ---
    dino_loss = DINOLoss(out_dim=768).to(device) # 768 для ViT-Base
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Цикл обучения ---
    print("Начало адаптации модели...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, crops in enumerate(data_loader):
            # `crops` - это список тензоров. Каждый элемент списка имеет размер [B, C, H, W]
            
            # Перемещаем все кропы на GPU
            crops = [c.to(device) for c in crops]
            
            # Разделяем на глобальные и локальные для удобства
            global_crops = crops[:2]
            local_crops = crops[2:]
            
            # --- Шаг Наставника (Teacher Forward Pass) ---
            with torch.no_grad(): # Градиенты не нужны
                teacher_output = teacher(torch.cat(global_crops))
            
            # --- Шаг Студента (Student Forward Pass) ---
            student_output = student(torch.cat(global_crops + local_crops))
            
            # --- Расчет Loss и обратное распространение ---
            loss = dino_loss(student_output, teacher_output)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- Обновление весов Наставника (EMA) ---
            # Моментум может меняться со временем
            current_momentum = EMA_MOMENTUM_START + (EMA_MOMENTUM_END - EMA_MOMENTUM_START) * (epoch / EPOCHS)
            update_teacher_ema(student, teacher, momentum=current_momentum)

        print(f"Эпоха [{epoch+1}/{EPOCHS}], Средняя Loss: {epoch_loss / len(data_loader):.4f}")

    end_time = time.time()
    print(f"Обучение завершено за {(end_time - start_time)/60:.2f} минут.")

    # --- Сохранение модели ---
    # Сохраняем модель СТУДЕНТА, так как она является конечным продуктом.
    torch.save(student.state_dict(), "dinov2_adapted_student.pth")
    print("Адаптированная модель студента сохранена в 'dinov2_adapted_student.pth'")