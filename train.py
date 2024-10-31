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


def train(model, model2, x, y, device_name) -> nn.Module: 
    torch.manual_seed(1)
    device = torch.device(device_name)
    x = torch.from_numpy(x).to(torch.float32).to(device).reshape(len(x), 1, 97)
    y_1 = pd.get_dummies(y).values
    y = torch.from_numpy(y_1).to(torch.float32).to(device)
    model = model.to(device)
    model2 = model2.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=4e-5)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    y1 = (train_y[:, 0] - train_y[:, 1] + 1).cpu().detach().numpy() / 2
    y2 = (test_y[:, 0] - test_y[:, 1] + 1).cpu().detach().numpy() / 2

    train_history = []
    test_history = []
    test_history_acc = []
    test_history_baseline = []
    train_history_acc = []
    train_history_acc_model_2 = []
    best_acc = 0
    count_since_best = 0

    for epoch in range(1000):
        ## train
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(train_y, outputs)
        loss.backward()
        optimizer.step()
        ans = ((outputs[:, 0] - outputs[:, 1]) >= 0).cpu().detach().numpy().astype(int)
        train_history_acc.append(1 - (sum(ans != y1) / len(y1)))

        train_history.append(loss.item())
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')


        ## test
        with torch.no_grad():
            outputs = model(test_x)
            outputs2 = model2(test_x)
            loss = criterion(test_y, outputs)
            test_history.append(loss.item())
            ans2 = ((outputs[:, 0] - outputs[:, 1]) >= 0).cpu().detach().numpy().astype(int)
            ans3 = ((outputs2[:, 0] - outputs2[:, 1]) >= 0).cpu().detach().numpy().astype(int)
            train_history_acc_model_2.append(1 - (sum(ans3 != y2) / len(y2)))
            test_history_acc.append(1 - (sum(ans2 != y2) / len(y2)))
            test_history_baseline.append(1 - (sum(y2 != 1) / len(y2)))
            if test_history_acc[-1] > best_acc:
                count_since_best = 0
                best_acc = test_history_acc[-1]
                torch.save(model.state_dict(), 'best_model.pkl')
            elif count_since_best == 50:
                model.load_state_dict(torch.load('best_model.pkl', weights_only=True))
                break
            else:
                count_since_best += 1
            if epoch % 20 == 0:
                print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')
                # print(train_history_acc_model_2[-1])
                # print(1 - (sum(ans2 != y2) / len(y2)))
                # print(1 - (sum(y2 != 1) / len(y2)))

    plt.title('accuracy')
    plt.plot(test_history_acc)
    plt.plot(test_history_baseline)
    plt.plot(train_history_acc)
    plt.legend(['test', 'baseline', 'train'])
    plt.savefig("training_acc.png")

    plt.cla()
    plt.plot(train_history)
    plt.plot(test_history)
    plt.grid(True)
    plt.title('Сходимость')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend(['train', 'test'])
    plt.savefig("training_error.png")
    
    return model


data = pd.read_parquet("./data/train.parquet")

arrs = []
data['mean'] = pd.NA
max_date = date(1999, 1, 1)
min_date = date(2025, 1, 1)
for index, row in data.iterrows():
    max_date = max(max_date, row['dates'][-1])
    min_date = min(min_date, row['dates'][0])

for index, row in data.iterrows():
    arr = row['values']
    if np.isnan(arr.min()):
        continue
    data.at[index, 'mean'] = np.mean(arr)
    date_start_diff = months(row['dates'][0], min_date)
    date_end_diff = months(max_date, row['dates'][-1])
    arr = np.pad(arr, (date_start_diff, date_end_diff), constant_values=(np.median(arr), np.median(arr)))
    arrs.append(arr) 

data = data.dropna()
vals = np.stack(arrs)

model = conv_model.ConvModel(hidden_size=150)
model2 = conv_model.ConvModel(hidden_size=150)
model2.load_state_dict(torch.load("./model.pkl", weights_only=True))
model = train(model, model2, vals, data['label'], 'cuda:0')
