# Построение траектории перемещения по данным монокулярной камеры мобильного телефона

## Полезные ссылки
[Отчетная преза](https://docs.google.com/presentation/d/1f3BKS4f2tlF1K5UaCT5WZelm-FVpcz17/edit?usp=sharing&ouid=100789703876834227826&rtpof=true&sd=true)
[Связанные статьи](https://docs.google.com/spreadsheets/d/1tRVtcsxwYK_F6P-zFTjac21a8KmPF63BA1GodeC9cLg/edit#gid=0)
[CV Themes Pitching](https://docs.google.com/presentation/d/1Nf0RiArBz9SlYQzwhTAr9tp0wdh8Jidl/edit#slide=id.g13b057c3d1d_0_35)
[Papers with Code Topic](https://paperswithcode.com/task/depth-and-camera-motion)

## Прогресс по задачам

### 6 июля
* Запущен бейзлайн SfmLearner, написан скрипт для визуализации траектории
* Скачаны KITTI Raw и KITTI Odometry
* ! Разобраться с визуализацией глубины
* ! Снять свои видео по плиткам и долгое во время прогулки

## Интринсики (нужно проверить)
ZB633KL: Sony IMX486 f=26mm, pix=1.25e-3mm, height=1080px, width=1920px
K = f/pix 0 width/2 0 f/pix height/2 0 0 1 = 20800 0 960 0 20800 540 0 0 1

