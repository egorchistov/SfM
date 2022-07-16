# Построение траектории перемещения по данным монокулярной камеры мобильного телефона

Репозиторий в разработке и будет дополнен

## Полезные ссылки

* [Presentation](https://docs.google.com/presentation/d/1E8qbHs3jCNF-ejv9MXRF8pTJ3mTutaLAZfMEL4c3ViQ/edit#slide=id.g1368b6a625b_0_2731)
* [Papers with Code Topic](https://paperswithcode.com/task/depth-and-camera-motion)

## Описание репозитория

* point_cloud — основной ноутбук для построения облака точек и определения масштаба по глубине
* calib — Ноутбук для калибровки камер и два набора изображений с разных камер
* notebooks — Google Colab ноутбуки для запуска репозиториев
* papers — pdf связанных статей
* path_examples — данные для построения траекторий с помощью visualize_path.py
* true_scales.ipynb — преобразование формата файлов масштаба
* visualize_path.py — визуализация траекторий
* gps2txt.py — конвертирование .gpx файлов OsmAnd в .txt файлы для visualize_path.py
* get_path.py — построение квадратной траектории для visualize_path.py

Пример команды для визуализации траектории

```shell
python visualize_path.py --npy=path_examples/phone_1_video_1_rate_5_sequence_1.npy --scale path_examples/phone_1_video_1_rate_5_sequence_1_5x_scales.npy --slam path_examples/coords_phone_1_video_1_rate_5_sequence_1_sift.npy --txt=path_examples/phone_1_video_1_rate_5_sequence_1.txt
```

