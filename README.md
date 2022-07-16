# Построение траектории перемещения по данным монокулярной камеры мобильного телефона

Репозиторий в разработке и будет дополнен

## Полезные ссылки

* [Presentation](https://docs.google.com/presentation/d/1E8qbHs3jCNF-ejv9MXRF8pTJ3mTutaLAZfMEL4c3ViQ/edit#slide=id.g1368b6a625b_0_2731)
* [Papers with Code Topic](https://paperswithcode.com/task/depth-and-camera-motion)

## Описание репозитория

* calib — Изображения и ноутбук для калибровки камеры
* notebook — Google Colab ноутбуки для запуска репозиториев
* papers — pdf связанных статей
* path_examples — данные для построения траекторий с помощью visualize_path.py и get_path.py
* point_cloud — ноутбук для построения облака точек и определения масштаба по глубине

Пример команды для визуализации траектории

```shell
python visualize_path.py --npy=path_examples/5.npy --txt=path_examples/5.txt
```

