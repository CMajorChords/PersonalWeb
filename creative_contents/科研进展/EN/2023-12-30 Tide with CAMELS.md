<h2 style='pointer-events: none;'>Use CAMELS to train a Tide model</h2>

Transformer structure has been applied in many fields, such as NLP and CV. However, in the time series field, whether the transformer is equally effective has been pointed out in literature that a dense encoder can achieve the same effect. Here, we use the [TiDE decoder]（ https://arxiv.org/abs/2304.08424 ）Compare with DLiner, NLiner, and Temporary Fusion Transformer to see if the transformer is effective in predicting runoff., The CAMELS dataset is used, consisting of 671 watersheds with a data length of 40 years. Each watershed contains runoff and meteorological data, including precipitation, temperature, radiation, evaporation, wind speed, etc. Here, precipitation, temperature, radiation, evaporation, wind speed are used as covariates, and runoff is used as the target variable

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
<h3 style='pointer-events: none;'>data processing</h3>

read data


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

Unlike autogluon, darts must extract data from each watershed, convert it to TimeSeries, and load it into a list.
Specify the column names for target and covariates


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

Extract target and covariates from the data


```python
# divide the data by watershed, and the divided data does not contain the watershed id
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
    ][static_cols] # static features of the watershed
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
# delete the data in the loop
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
<h3 style='pointer-events: none;'>Model parameters</h3>

* Gradient clipping: Prevent gradient explosion by setting an upper limit for gradients
* Learning rate: Most of the abilities of the model are learned in the early stages of training, so in the early stages of training, we can set a larger learning rate to enable the model to learn quickly. Then, in the later stages of training, we can set a smaller learning rate to make the model more stable.
* Early stopping: By monitoring the loss of the validation set, stop training when the loss no longer decreases to prevent overfitting of the model.


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
# learning rate decay
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR # or ReduceLROnPlateau
lr_scheduler_kwargs = {
    "gamma": 0.999,
}
# early stopping
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
    "likelihood": None, # GaussianLikelihood() or StudentTLikelihood()
    "save_checkpoints": True, # save model
    "force_reset": True, # reset model
    "batch_size": 512,
    "random_state": 3407,
    "work_dir": "models",
    "n_epochs": 500,
    "log_tensorboard": True,
    # "dropout": 0.1,
    # "hidden_size": 64,
}
```
<h3 style='pointer-events: none;'>Model configuration</h3>

Most models use the same parameters, the only exception is that Tide has tried whether Reverible Instance Normalization is used


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
# train
for name, model in models.items():
    
    # early stopping
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
