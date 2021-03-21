# CoinPy
CoinPy is an open-source project for making cryptocurrency available for everyone.

## Installation
```bash
pip install coin-py
```
Installation from source
```bash
git clone https://github.com/Demirrr/coin-py
# To develop
pip install -e .[dev]
python -m pytest tests
```

## Examples
```python
import coinpy as cp
# Fetch and Store all coins provided in Coinbase
cp.Collector().fetch_and_save()
```

### Todos

- [ ] Provide previous information about coins.
- [ ] Examples of using coinpy within [sktime](https://github.com/alan-turing-institute/sktime).
- [ ] Adapt test driven software development.
- [ ] Integrate crontab for automatic scrapping.

### In Progress

- [ ] Examples of using coinpy within [sktime](https://github.com/alan-turing-institute/sktime).
- [ ] Integrate crontab for automatic scrapping.

### Done ✓

- [x] Create fetching coins from Coinbase.