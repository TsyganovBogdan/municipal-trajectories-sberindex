# Источники

Исследование использует данные о потреблении за 2023–2024 годы и транспортных расстояниях и доступности рынков на конец 2024 года. Расчёты и редакция материалов от 7 октября 2026 года.

## Данные

1. СберИндекс. [Описание муниципального набора](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim). Состав, методология показателей и условия использования.
2. СберИндекс. [Архив hackathonlicence.zip](https://sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip). Использованы `consumption.parquet`, `connection.parquet`, `market_access.parquet`. Исходные файлы и документ с лицензией включены в `data/raw/`; контрольные суммы приведены в `results/data_audit.json`.
3. YuliyaTorgasheva/spo_map. [Открытая карта муниципалитетов](https://github.com/YuliyaTorgasheva/spo_map), [municipalities.geojson](https://raw.githubusercontent.com/YuliyaTorgasheva/spo_map/master/data/municipalities.geojson). Справочник получен 3 октября 2026 года. Для 2 016 ID извлечены поля `territory_id`, `region_name`, `municipal_district_name_short`, `municipal_district_type`, `oktmo_base`. Сопоставление однозначно. Названия используются в подписях и поиске; в признаки модели они не входят. Справочник может отражать административное устройство более позднего периода. Геометрия и полный GeoJSON в состав проекта не включены.

## Методы

4. Shalileh S., Antonov E. A., Tsyplakova D. A. Публикация об оценке качества кластеризации (2025). [DOI: 10.1134/S1064562425700589](https://doi.org/10.1134/S1064562425700589). Индексы AVI и AVU реализованы по формулам 18–21 с обобщением бинарной смежности на неотрицательные веса. Определения приведены в разделе 4 отчёта.
5. Traag V. A., Waltman L., van Eck N. J. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports 9, 5233 (2019). [DOI: 10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z). [Препринт](https://arxiv.org/abs/1810.08473).
6. [Документация leidenalg](https://leidenalg.readthedocs.io/en/stable/). Используется `RBConfigurationVertexPartition`. Кластеризация выполняется отдельно для каждого месяца.
7. [Документация scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html). Реализации K-means, Ward, silhouette и Calinski–Harabasz.
8. Halkidi M., Vazirgiannis M. Clustering validity assessment: finding the optimal partitioning of a data set. ICDM 2001. [DOI: 10.1109/ICDM.2001.989517](https://doi.org/10.1109/ICDM.2001.989517). Реализация S_Dbw использует популяционную дисперсию и подсчёт точек внутри радиуса. Обработка нулевых знаменателей описана в отчёте и `src/metrics.py`.

## Конкурс

9. СберИндекс. [Страница конкурса исследовательских проектов](https://sber.ru/sberindex/konkurs_sberindex).
10. СберИндекс. [Положение о конкурсе](https://sber.ru/common/assets/sber/sberindex_polozhenie_o_konkurse.pdf).

Диаграммы, таблицы и интерактивная проекция построены по результатам расчётов проекта. Условия распространения данных приведены в `DATA_LICENSE.md`.
