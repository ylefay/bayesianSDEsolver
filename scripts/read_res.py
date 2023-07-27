import numpy as np
import pandas as pd

res_cov = pd.read_csv('./res_cov.csv')
for i in range(0, 24):
    if i != 0:
        res = res_cov[f'0.0.{str(i)}'].dropna()
    else:
        res = res_cov[f'0.0'].dropna()
    l = len(res)
    res.index += 1
    res = pd.concat([pd.Series([0.0]), res])
    linspace = np.linspace(0, 1, l+1)
    res = {'t': linspace, 'res': res}
    df = pd.DataFrame.from_dict(data=res)
    df.to_csv(f'./res_cov_{l}.csv', header=True)