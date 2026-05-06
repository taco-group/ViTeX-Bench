"""ViTeX-Bench evaluation package.

The scripts under ``benchmark/`` are standalone CLIs, each inserting its
own directory into ``sys.path`` so the sibling modules (``bench_utils``,
``text_score``, ...) resolve as bare imports.

CLI entry points:

* ``benchmark/ocr_extract.py`` — PP-OCRv5 over source and prediction videos (CPU).
* ``benchmark/evaluate.py`` — GPU 13-metric pipeline.

Importable modules: ``text_metrics``, ``visual_metrics``, ``locality_metrics``,
``text_score``, ``bench_utils``, ``lang_detect``. Import them directly via
``sys.path.insert(0, '.../ViTeX-Bench/benchmark'); import text_metrics``
rather than ``from benchmark import text_metrics``.
"""
