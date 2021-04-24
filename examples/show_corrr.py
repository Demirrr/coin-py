from coinpy import Analyser
from coinpy.util import *
import pandas as pd
import matplotlib.pyplot as plt

analyser = Analyser(path_correlation='../ProcessedData/corr_coin.json')

n = 3
for i in analyser.max_n_corr(n):
    print(i)
print('#' * 10)
for i in analyser.min_n_corr(n):
    print(i)
