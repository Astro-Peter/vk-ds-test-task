import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch
from datetime import date
import conv_model

def months(d1, d2):
    return d1.month - d2.month + 12*(d1.year - d2.year)

data = pd.read_parquet("./data/train.parquet")
print(data.columns)
data['mean'] = np.nan
arrs = []
max_date = date(1999, 1, 1)
min_date = date(2025, 1, 1)
for index, row in data.iterrows():
    max_date = max(max_date, row['dates'][-1])
    min_date = min(min_date, row['dates'][0])

for index, row in data.iterrows():
    if np.isnan(row['values'].min()):
        continue
    arr = row['values']
    data.at[index, 'mean'] = np.mean(arr)
    date_start_diff = months(row['dates'][0], min_date)
    date_end_diff = months(max_date, row['dates'][-1])
    arr = np.pad(arr, (date_start_diff, date_end_diff), constant_values=(data.at[index, 'mean'], data.at[index, 'mean']))
    arrs.append(arr) 

vals = np.stack(arrs)
data = data.dropna()

model = conv_model.ConvModel(hidden_size=150)
model.load_state_dict(torch.load("./model.pkl", weights_only=False))
model.eval()
ans = model(torch.from_numpy(vals).to(torch.float32).reshape(len(arrs), 1, 97)).cpu().detach().numpy()
ans2 = ((ans[:, 0] - ans[:, 1]) >= 0).astype(int)
ans3 = -ans2 + 1 
y2 = (pd.get_dummies(data['label']).values[:, 0] ^ pd.get_dummies(data['label']).values[:, 1] + 1) / 2

print(sum(ans3 == data['label']) / len(ans3))
print(sum(ans3))
answers = pd.DataFrame(data={'answers': ans3})
answers.to_csv("./data/answers.csv")