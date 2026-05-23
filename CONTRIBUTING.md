# Contributing guide

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/your-username/skin_disease_detection.git
cd skin_disease_detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Aim for 70%+ coverage before opening a PR.

## Code style

```bash
flake8 src/ --max-line-length=100
mypy src/ --ignore-missing-imports
```

Both must pass (zero errors) before a PR can be merged.

## Training the model

1. Place raw images in `raw_images/<class_name>/`
2. Run `python src/auto_filter.py` to clean the dataset
3. Run `python src/organize_dataset.py` to split into train/val/test
4. Run `python src/train_model.py` to train
5. Model saved to `model/best_model.h5`

## Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Linting passes (`flake8 src/`)
- [ ] Docstrings updated for any changed functions
- [ ] CHANGELOG.md entry added
- [ ] No new dead code or unused imports

## Reporting bugs

Open a GitHub Issue with:
- OS and Python version
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output from `logs/app.log`
