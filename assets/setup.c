[metadata]
license = MIT

[options]
zip_safe = False
include_package_data = True
packages = find:

[options.entry_points]
console_scripts =
    algerian-forest-fire-predictor = src.pipeline.train_pipeline:main

[tool:pytest]
addopts = --verbose --cov=src --cov-report=term-missing