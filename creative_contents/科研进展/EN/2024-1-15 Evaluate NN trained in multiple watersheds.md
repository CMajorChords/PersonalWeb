<h2 style='pointer-events: none;'>Evaluate networks trained in multiple watersheds</h2>

for example, TiDE model, NLinear model, DLinear model, TFT model.

```python
%matplotlib inline

import torch
import numpy as np
import pandas as pd
import shutil

from darts.models import TiDEModel, TFTModel, NLinearModel, DLinearModel
from darts.utils.model_selection import train_test_split
from darts.dataprocessing.transformers.scaler import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.metrics import mae, mse
from darts import TimeSeries

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)
```

    D:\Software\miniconda3\Lib\site-packages\dask\dataframe\_pyarrow_compat.py:17: FutureWarning: Minimal version of pyarrow will soon be increased to 14.0.1. You are using 11.0.0. Please consider upgrading.
      warnings.warn(
    

test data preparation


```python
rainfall_source = "daymet"
varying_features = pd.read_csv(
    f"../data/varying_features_{rainfall_source}_preprocessed.csv",
    dtype={"gauge_id": str},
    parse_dates=True,
)
static_features = pd.read_csv(
    f"../data/static_features_preprocessed.csv",
    dtype={"gauge_id": str},
    parse_dates=True,
)
if rainfall_source == "daymet":
    covariates_cols = [
        "prcp(mm/day)",
        "srad(W/m2)",
        "tmax(C)",
        "tmin(C)",
        "vp(Pa)",
        "Year",
        "Mnth_sin",
        "Mnth_cos",
        "Day_sin",
        "Day_cos",
    ]
else:
    covariates_cols = [
        "PRCP(mm/day)",
        "SRAD(W/m2)",
        "Tmax(C)",
        "Tmin(C)",
        "Vp(Pa)",
        "Year",
        "Mnth_sin",
        "Mnth_cos",
        "Day_sin",
        "Day_cos",
    ]
target_cols = [
    "runoff"
]
static_cols = pd.read_excel(
    "../data/static features.xlsx",
)["features"].tolist()
# 以流域为单位划分数据，划分后的数据不含流域id
varying_features_grouped = varying_features.groupby(
    "gauge_id",
    as_index=False,
    group_keys=False,
)
# list of TimeSeries
covariates_list = []
train_target_list = []
val_target_list = []
test_target_list = []
test_time_length = 365 * 3
val_time_length = 365 * 4
for gauge_id, gauge_data in varying_features_grouped:
    # static features
    static_features_gauge = static_features[
        static_features["gauge_id"] == gauge_id
    ][static_cols] # 从static_features中提取出该流域的静态特征
    # covariates
    covariates_timeseries = TimeSeries.from_dataframe(
        gauge_data,
        time_col="datetime",
        value_cols=covariates_cols,
    )
    covariates_list.append(covariates_timeseries)
    # test target
    test_target_timeseries = TimeSeries.from_dataframe(
        gauge_data.iloc[-test_time_length:],
        time_col="datetime",
        value_cols=target_cols,
        static_covariates=static_features_gauge,
    )
    test_target_list.append(test_target_timeseries)
    # val target
    val_target_timeseries = TimeSeries.from_dataframe(
        gauge_data.iloc[-(test_time_length + val_time_length):-test_time_length],
        time_col="datetime",
        value_cols=target_cols,
        static_covariates=static_features_gauge,
    )
    val_target_list.append(val_target_timeseries)
    # train target
    train_target_timeseries = TimeSeries.from_dataframe(
        gauge_data.iloc[:-(
            test_time_length + val_time_length)],
        time_col="datetime",
        value_cols=target_cols,
        static_covariates=static_features_gauge,
    )
    train_target_list.append(train_target_timeseries)
# 删去循环中的临时变量
del (
static_features_gauge,
covariates_timeseries,
test_target_timeseries,
val_target_timeseries,
train_target_timeseries,
gauge_data,
gauge_id,
)
```

read model from checkpoint


```python
# 如果有GPU，使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 读取模型
model_tide = TiDEModel.load_from_checkpoint(
    model_name="tide_normal",
    work_dir="../models",
    best=True,
    map_location=device,
)
```

test on one basin


```python
pred_steps = 1
input_chunk_length = 24
# 测试集
test_id = 0
test_target = test_target_list[test_id]
predict_target = test_target[:input_chunk_length+pred_steps]
for i in range(len(test_target_list)-input_chunk_length-pred_steps-7):
    model_input = test_target[i:i+input_chunk_length]
    model_output = model_tide.predict(
        n=pred_steps,
        series=model_input,
        future_covariates=covariates_list[test_id],
        n_jobs=6,
    )
    predict_target.append(model_output)
fig, ax = plt.subplots(figsize=(20, 10))
test_target[:-120].plot(label="true", ax=ax)
predict_target[-120].plot(label="predict", ax=ax)
```
restore normalized data
```
import pickle
with open("data/varying_features_daymet_MinMaxScaler.pickle", "rb") as f:
    varying_features_scaler = pickle.load(f)
with open("data/static_features_MinMaxScaler.pickle", "rb") as f:
    static_features_scaler = pickle.load(f)
# inverse transform
predict_target = predict_target.pd_dataframe()
test_target = test_target.pd_dataframe()
#%% Scaler has six columns, but predict_ target and test_ target has only one column, so only the first column is taken
from sklearn.preprocessing import MinMaxScaler
i = 5
single_col_scaler = MinMaxScaler()
single_col_scaler.min_, single_col_scaler.scale_ = varying_features_scaler.min_[i], varying_features_scaler.scale_[i]
predict_target = single_col_scaler.inverse_transform(predict_target)
test_target = single_col_scaler.inverse_transform(test_target)
#%% 还原log
predict_target = np.exp(predict_target) - 1
test_target = np.exp(test_target) - 1
#%% 取test_target的和predict_target的共同部分
```
draw a picture to see
```
fig, ax = plt.subplots(figsize=(20, 10))
series_length = min(len(test_target), len(predict_target))
ax.plot(
    test_target[:365],
    label="test_target",
)
ax.plot(
    predict_target[:365],
    label="predict_target",
)
ax.legend()
plt.show()
```
calculate NSE
```
#%% 计算两列数据的Nash-Sutcliffe Efficiency
def nse(obs, sim):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
print(nse(test_target[:365], predict_target[:365]))
```