<h2 style='pointer-events: none;'>Use Temporal fusion transformer to predict runoff</h2>

The Temporal Fusion Transformer (TFT) is a deep learning model for time series forecasting. It was introduced in the paper [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister, and Anelia Angelova. The model is based on the Transformer architecture and combines ideas from RNNs and feed-forward networks. It is designed to capture long-term dependencies in time series and to allow for multi-horizon forecasting. The model is also interpretable, meaning that it can be used to understand the importance of each input feature for the prediction.This notebook will give a brief introduction to the TFT model and use it to predict the runoff of a specific basin in the CAMELS dataset.
<h3 style='pointer-events: none;'>1. import libraries</h3>

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from darts import  TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings

warnings.filterwarnings("ignore")
import  logging

logging.disable(logging.CRITICAL) # darts is very verbose, so we turn off logging for now
```
<h3 style='pointer-events: none;'>2. Data processing</h3>

We use CAMELS class to read data from CAMLES dataset. The CAMELS dataset contains hydrological, meteorological, and geomorphological data for 671 catchments in the United States. The data is organized in 7 files, each containing different attributes of the catchments. The CAMELS class reads the data from these files and stores them in a dictionary. The get_basin_data method of the class returns a dictionary containing all the attributes of a given basin. The attributes include climate, geology, hydrology, soil, topography, vegetation, and the name of the basin. The class also reads the rainfall and runoff data for each basin and stores them in the dictionary. The rainfall data is stored in a pandas DataFrame, and the runoff data is stored in a Darts TimeSeries object. The get_basin_data method also returns the area of the basin. The area is stored in the third line of the rainfall data file.


```python
class CAMELS():
    """
    CAMELS data loader
    """

    def __init__(self, path: str):
        """
        read CAMELS data from the given path, and store basin attributes in the class
        :param path: the path to the CAMELS data folder
        """
        self.data_path = path
        self.attributes = ["clim", "geol", "hydro", "name", "soil", "topo", "vege"]
        self.clim = pd.read_csv(f"{path}/camels_clim.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str},
                                sep=";",
                                )
        self.geol = pd.read_csv(f"{path}/camels_geol.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str},
                                sep=";",
                                )
        self.hydro = pd.read_csv(f"{path}/camels_hydro.txt",
                                 index_col="gauge_id",
                                 dtype={"gauge_id": str},
                                 sep=";",
                                 )
        self.name = pd.read_csv(f"{path}/camels_name.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str, "huc_02": str},
                                sep=";",
                                )
        self.soil = pd.read_csv(f"{path}/camels_soil.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str},
                                sep=";",
                                )
        self.topo = pd.read_csv(f"{path}/camels_topo.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str},
                                sep=";",
                                )
        self.vege = pd.read_csv(f"{path}/camels_vege.txt",
                                index_col="gauge_id",
                                dtype={"gauge_id": str},
                                sep=";",
                                )

    def get_basin_data(self, gauge_id, rainfall_data_source="daymet", datetime_reserve=False) -> dict:
        """
        get all basin attributes for a given basin id
        :param gauge_id:  the id of the basin
        :return:  a dictionary containing all basin attributes and rainfall-runoff data
        """
        basin = {}
        # basin attributes
        for attr in self.attributes:
            basin[attr] = getattr(self, attr).loc[gauge_id]

        # rainfall_data_source can be either a string or an integer
        if isinstance(rainfall_data_source, int):
            rainfall_data_sources = ["daymet", "maurer", "nldas"]
            rainfall_data_source = rainfall_data_sources[rainfall_data_source]
        
        # set the path to the rainfall data
        huc_02 = self.name.loc[gauge_id]["huc_02"]  # get the huc_02 code
        rainfall_data_path = f"{self.data_path}/basin_mean_forcing/{rainfall_data_source}/{huc_02}/{gauge_id}_lump_cida_forcing_leap.txt"
        
        # set the rainfall attribute
        rainfall_data = pd.read_csv(rainfall_data_path, 
                                    skiprows=3, 
                                    sep="\s+")
        rainfall_data.drop(columns=["Hr", "dayl(s)", "swe(mm)"], axis=1, inplace=True)
        rainfall_data.index = pd.to_datetime(rainfall_data[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "Month"}))
        basin["rainfall"] = rainfall_data

        # set the area attribute
        with open(rainfall_data_path, "r") as f:
            # The area is stored in the third line of the file
            basin["area"] = float(f.readlines()[2])

        # set the runoff attribute
        runoff_data_path = f"{self.data_path}/usgs_streamflow/{huc_02}/{gauge_id}_streamflow_qc.txt"
        runoff_data = pd.read_csv(runoff_data_path,
                                  header=None,
                                  names=["gauge_id", "Year", "Mnth", "Day", "runoff", "qc"],
                                  sep="\s+",
                                  )
        # create a datetime index from the year, month, and day columns
        runoff_data.index = pd.to_datetime(runoff_data[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "Month"}))
        # # if qc is M, it means that the data is missing,use linear interpolation to fill the missing data
        runoff_data.loc[runoff_data["qc"] == "M", "runoff"] = np.nan
        runoff_data["runoff"] = runoff_data["runoff"].interpolate()
        basin["runoff"] = runoff_data

        # concatenate the rainfall and runoff data
        basin["data"] = pd.concat([rainfall_data, runoff_data["runoff"]], axis=1)
        # if datetime data is not needed, drop the datetime column
        if not datetime_reserve:
            basin["data"].drop(columns=["Year", "Mnth", "Day"], inplace=True)
        return basin

camels = CAMELS("../data")
basin_id = "01013500"
basin = camels.get_basin_data(basin_id, datetime_reserve=True)
basin["data"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Mnth</th>
      <th>Day</th>
      <th>prcp(mm/day)</th>
      <th>srad(W/m2)</th>
      <th>tmax(C)</th>
      <th>tmin(C)</th>
      <th>vp(Pa)</th>
      <th>runoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-01-01</th>
      <td>1980</td>
      <td>1</td>
      <td>1</td>
      <td>0.00</td>
      <td>153.40</td>
      <td>-6.54</td>
      <td>-16.30</td>
      <td>171.69</td>
      <td>655.0</td>
    </tr>
    <tr>
      <th>1980-01-02</th>
      <td>1980</td>
      <td>1</td>
      <td>2</td>
      <td>0.00</td>
      <td>145.27</td>
      <td>-6.18</td>
      <td>-15.22</td>
      <td>185.94</td>
      <td>640.0</td>
    </tr>
    <tr>
      <th>1980-01-03</th>
      <td>1980</td>
      <td>1</td>
      <td>3</td>
      <td>0.00</td>
      <td>146.96</td>
      <td>-9.89</td>
      <td>-18.86</td>
      <td>138.39</td>
      <td>625.0</td>
    </tr>
    <tr>
      <th>1980-01-04</th>
      <td>1980</td>
      <td>1</td>
      <td>4</td>
      <td>0.00</td>
      <td>146.20</td>
      <td>-10.98</td>
      <td>-19.76</td>
      <td>120.06</td>
      <td>620.0</td>
    </tr>
    <tr>
      <th>1980-01-05</th>
      <td>1980</td>
      <td>1</td>
      <td>5</td>
      <td>0.00</td>
      <td>170.43</td>
      <td>-11.29</td>
      <td>-22.21</td>
      <td>117.87</td>
      <td>605.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2014-12-27</th>
      <td>2014</td>
      <td>12</td>
      <td>27</td>
      <td>0.00</td>
      <td>103.01</td>
      <td>2.15</td>
      <td>-2.55</td>
      <td>508.64</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>2014-12-28</th>
      <td>2014</td>
      <td>12</td>
      <td>28</td>
      <td>2.79</td>
      <td>104.63</td>
      <td>2.46</td>
      <td>-3.70</td>
      <td>461.44</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>2014-12-29</th>
      <td>2014</td>
      <td>12</td>
      <td>29</td>
      <td>0.02</td>
      <td>193.62</td>
      <td>-0.76</td>
      <td>-16.03</td>
      <td>175.39</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>2014-12-30</th>
      <td>2014</td>
      <td>12</td>
      <td>30</td>
      <td>0.00</td>
      <td>180.57</td>
      <td>-13.31</td>
      <td>-23.54</td>
      <td>90.01</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>2014</td>
      <td>12</td>
      <td>31</td>
      <td>0.00</td>
      <td>185.32</td>
      <td>-14.84</td>
      <td>-25.60</td>
      <td>80.00</td>
      <td>205.0</td>
    </tr>
  </tbody>
</table>
<p>12784 rows × 9 columns</p>
</div>



create covariates and timeseries


```python
series = TimeSeries.from_series(basin["data"]["runoff"])
covariates = TimeSeries.from_dataframe(basin["data"].drop(columns=["runoff"]))
covariates.plot()
```




    <Axes: xlabel='time'>




    
    


normalization and split


```python
training_cutoff = pd.Timestamp("2011-01-01")
train, val = series.split_after(training_cutoff)

transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

scaler_covs = Scaler()
cov_train, cov_val = covariates.split_after(training_cutoff)
scaler_covs.fit(cov_train)
covariates_transformed = scaler_covs.transform(covariates)
```
<h3 style='pointer-events: none;'>3. Create a model</h3>

In this model, we use quantile forecast to predict runoff


```python
quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
input_chunk_length = 24
forecast_horizon = 7
my_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=300,
    add_relative_index=False,
    add_encoders=None,
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  # QuantileRegression is set per default
    # loss_fn=MSELoss(),
    random_state=42,
)
```
<h3 style='pointer-events: none;'>4. Train the model</h3>

```python
my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)
```


    Training: |          | 0/? [00:00<?, ?it/s]





    TFTModel(hidden_size=64, lstm_layers=2, num_attention_heads=4, full_attention=False, feed_forward=GatedResidualNetwork, dropout=0.1, hidden_continuous_size=8, categorical_embedding_sizes=None, add_relative_index=False, loss_fn=None, likelihood=QuantileRegression(quantiles: Optional[List[float]] = None), norm_type=LayerNorm, use_static_covariates=True, input_chunk_length=24, output_chunk_length=7, batch_size=16, n_epochs=300, add_encoders=None, random_state=42)



Look at predictions on the validation set


```python
# before starting, we define some constants
num_samples = 300

figsize = (9, 6)
lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

def eval_model(model, n, actual_series, val_series):
    pred_series = model.predict(n=n, num_samples=num_samples)

    # plot actual series
    plt.figure(figsize=figsize)
    actual_series[: pred_series.end_time()].plot(label="actual")

    # plot prediction with quantile ranges
    pred_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

    plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
    plt.legend()


eval_model(my_model, 24, series_transformed, val_transformed)
```


    Predicting: |          | 0/? [00:00<?, ?it/s]



    
    



```python
backtest_series = my_model.historical_forecasts(
    series_transformed,
    future_covariates=covariates_transformed,
    start=train.end_time() + train.freq,
    num_samples=num_samples,
    forecast_horizon=forecast_horizon,
    stride=forecast_horizon,
    last_points_only=False,
    retrain=False,
    verbose=True,
)

def eval_backtest(backtest_series, actual_series, horizon, start, transformer):
    plt.figure(figsize=figsize)
    actual_series.plot(label="actual")
    backtest_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    backtest_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    plt.legend()
    plt.title(f"Backtest, starting {start}, {horizon}-months horizon")
    print(
        "MAPE: {:.2f}%".format(
            mape(
                transformer.inverse_transform(actual_series),
                transformer.inverse_transform(backtest_series),
            )
        )
    )


eval_backtest(
    backtest_series=concatenate(backtest_series),
    actual_series=series_transformed,
    horizon=forecast_horizon,
    start=train.start_time(),
    transformer=transformer,
)

```


    Predicting: |          | 0/? [00:00<?, ?it/s]


    MAPE: 9.49%
    


    
    


save the model


```python
my_model.save(f"../models/TFT_{basin_id}.pkl")
```
