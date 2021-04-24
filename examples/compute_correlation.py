import pandas as pd
from coinpy.util import *
import json
pd.options.display.max_columns = None

res = dict()
for i in get_all_folders('../ProcessedData'):
    if 'csv' in i :
        df = pd.read_csv(i, index_col='time', parse_dates=True)
        suffix = i[i.rfind('/') + 1:-4]
        res[suffix] = df[['low_' + suffix, 'high_' + suffix]].mean(axis=1)
corr_res = dict()
seen = set()
for k, v in res.items():
    for kk, vv in res.items():
        if k == kk:
            continue
        if k + '_' + kk in seen:
            continue

        seen.add(k + '_' + kk)
        seen.add(kk + '_' + k)
        corr_res[k + '_' + kk] = v.corr(vv)

with open("../ProcessedData/corr_coin.json", "w") as outfile:
    json.dump(corr_res, outfile, indent=4)