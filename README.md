# Построение траектории перемещения по данным монокулярной камеры мобильного телефона

## Полезные ссылки
[Отчетная преза](https://docs.google.com/presentation/d/1f3BKS4f2tlF1K5UaCT5WZelm-FVpcz17/edit?usp=sharing&ouid=100789703876834227826&rtpof=true&sd=true)
[Backup отчетной презы](https://docs.google.com/presentation/d/1E8qbHs3jCNF-ejv9MXRF8pTJ3mTutaLAZfMEL4c3ViQ/edit#slide=id.g1368b6a625b_0_2731)
[Связанные статьи](https://docs.google.com/spreadsheets/d/1tRVtcsxwYK_F6P-zFTjac21a8KmPF63BA1GodeC9cLg/edit#gid=0)
[CV Themes Pitching](https://docs.google.com/presentation/d/1Nf0RiArBz9SlYQzwhTAr9tp0wdh8Jidl/edit#slide=id.g13b057c3d1d_0_35)
[Papers with Code Topic](https://paperswithcode.com/task/depth-and-camera-motion)

## Прогресс по задачам

### 05.07 — Сделано
* Прочитали статью SfMLearner и запустили инференс
* Записали несколько коротких видео и построили траекторию
* Нашли приблизительную калибровку камеры по метаданным
* Скачали KITTI и Cityscapes, прогнали инференс на Cityscapes

### 06.07 — Сделано
* Запустили SfMLearner, написали скрипт для построения траектории
* Записали свое видео с прохождением известной траектории
* Пострили траектории для KITTI и для своего набора данных

### 07.07 — Сделано
* Записали длинный ролик для обучения на улицах города 
* Откалибровали камеру методом шахматной доски
* Сделали замеры скорости движения, произвели сверку данных скорости с реальными
* Произвели попытку отобразить облако точек для траектории движения

### Далекие планы
* Анализ других архитектур PoseNet, DepthNet и процессов обучения
* Дообучение модели на своих данных (семплирование со скоростью KITTI)

### Понять в коде
* Понять как считается scale_factor в test_pose
* Понять train validate_with_gt_pose как формируется final_pose

