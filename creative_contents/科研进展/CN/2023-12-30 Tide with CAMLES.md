<h2 style='pointer-events: none;'>使用CAMELS数据训练Tide模型</h2>

transformer结构在很多领域都有应用，比如NLP，CV，但是在时间序列领域，transformer是否同样有效，有文献指出一个dense encoder就可以达到同样的效果，这里使用[TiDE解码器](https://arxiv.org/abs/2304.08424)和DLinear、NLinear、Temporal Fusion Transformer进行对比，看看在径流预测上，transformer是否有效。，数据集使用CAMELS数据集，包含671个流域，每个流域的数据长度为40年，每个流域的数据包含径流和气象数据，气象数据包含降水、温度、辐射、蒸发、风速等，这里使用降水、温度、辐射、蒸发、风速作为协变量，径流作为目标变量.


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
<h3 style='pointer-events: none;'>数据处理</h3>

读取数据


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
```

与autogluon不一样，darts必须将每个流域的数据提取出来，然后转换成TimeSeries格式，再装入一个list中
指定target和covariates的列名


```python
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
```

将target和covariates提取出来


```python
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
    test_time_length = 365 * 3
    test_target_timeseries = TimeSeries.from_dataframe(
        gauge_data.iloc[-test_time_length:],
        time_col="datetime",
        value_cols=target_cols,
        static_covariates=static_features_gauge,
    )
    test_target_list.append(test_target_timeseries)
    # val target
    val_time_length = 365 * 4
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
<h3 style='pointer-events: none;'>模型参数设置</h3>

* Gradient clipping: 通过为梯度设置上限，防止梯度爆炸
* Learning rate: 模型的大部分能力都是在训练的前期学习到的，所以在训练的前期，我们可以设置一个较大的学习率，让模型快速学习，然后在训练后期，我们可以设置一个较小的学习率，让模型更加稳定。
* Early stopping: 通过监控验证集的loss，当loss不再下降时，停止训练，防止模型过拟合。


```python

optimizer_kwargs = {
    "lr": 1e-3,
}
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 1000,
    "accelerator": "auto",
    "callbacks": [],
}
# 学习率衰减
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR # or ReduceLROnPlateau
lr_scheduler_kwargs = {
    "gamma": 0.999,
}
# 早停
early_stopping_args = {
    "monitor": "val_loss",
    "patience": 30,
    "min_delta": 1e-3,
    "mode": "min",
}

common_model_args = {
    "input_chunk_length": 24,
    "output_chunk_length": 7,
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None, # 使用概率分布预测
    "save_checkpoints": True, # 保存模型
    "force_reset": True, # 重置模型
    "batch_size": 512,
    "random_state": 3407,
    "work_dir": "models",
    "n_epochs": 500,
    "log_tensorboard": True,
    # "dropout": 0.1,
    # "hidden_size": 64,
}
```
<h3 style='pointer-events: none;'>模型配置</h3>

大多数模型使用的是相同的参数，唯一例外的是Tide在是否Reverible Instance Normalization都进行了尝试


```python
model_DLinear = DLinearModel(
    **common_model_args,
    model_name="DLinear",
    use_static_covariates=False
)
model_NLinear = NLinearModel(
    **common_model_args,
    model_name="NLinear",
    use_static_covariates=False
)
model_tide = TiDEModel(**common_model_args, 
                       model_name="tide_normal",
                       use_reversible_instance_norm=True,
                       hidden_size=64,
                       dropout=0.1,
                       )
model_tide_rin = TiDEModel(
    **common_model_args,
    model_name="tide_rin",
    use_reversible_instance_norm=True,
    hidden_size=64,
    dropout=0.1,
)
model_tft = TFTModel(**common_model_args,
                     model_name="tft",
                     hidden_size=64,
                     dropout=0.1,
                     )
models = {
    "TiDE+RIN": model_tide_rin,
    "TiDE": model_tide,
    "TFT": model_tft,
    "DLinear": model_DLinear,
    "NLinear": model_NLinear,
}
```


```python
# 训练模型并且从检查点中加载最佳模型
for name, model in models.items():
    
    # early stopping需要在每次训练前重置
    pl_trainer_kwargs["callbacks"] = [
        EarlyStopping(
            **early_stopping_args,
        )
    ]
    
    model.fit(
        series=train_target_list,
        future_covariates=covariates_list,
        val_series=val_target_list,
        val_future_covariates=covariates_list,
        verbose=True,
    )
```
