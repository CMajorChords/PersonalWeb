<h2 style='pointer-events: none;'>Features Preprocessing</h2>

This notebook is used to generate features for the model training.
Dataset is [CAMELS](https://ral.ucar.edu/solutions/products/camels) dataset, which is a large dataset of hydrological variables for 671 catchments across the contiguous United States.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
%matplotlib inline
```

___
<h3 style='pointer-events: none;'>1. Data preparation</h3>

Firstly, we need to prepare the data for tsfresh. The data should be in the format of pandas dataframe.


```python
rainfall_source = "daymet" # choose the rainfall source, daymet, maurer, or nldas
varying_features = pd.read_csv(
    f'../data/varying_features_{rainfall_source}_interpolate.csv',
    dtype={"gauge_id": str},
    parse_dates=["datetime"],
)
static_features = pd.read_csv(
    '../data/static_features.csv',
    dtype={"gauge_id": str},
) 
varying_features
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
      <th>datetime</th>
      <th>gauge_id</th>
      <th>runoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>1</td>
      <td>1</td>
      <td>0.00</td>
      <td>153.40</td>
      <td>-6.54</td>
      <td>-16.30</td>
      <td>171.69</td>
      <td>1980-01-01</td>
      <td>01013500</td>
      <td>655.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>1</td>
      <td>2</td>
      <td>0.00</td>
      <td>145.27</td>
      <td>-6.18</td>
      <td>-15.22</td>
      <td>185.94</td>
      <td>1980-01-02</td>
      <td>01013500</td>
      <td>640.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>1</td>
      <td>3</td>
      <td>0.00</td>
      <td>146.96</td>
      <td>-9.89</td>
      <td>-18.86</td>
      <td>138.39</td>
      <td>1980-01-03</td>
      <td>01013500</td>
      <td>625.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1980</td>
      <td>1</td>
      <td>4</td>
      <td>0.00</td>
      <td>146.20</td>
      <td>-10.98</td>
      <td>-19.76</td>
      <td>120.06</td>
      <td>1980-01-04</td>
      <td>01013500</td>
      <td>620.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1980</td>
      <td>1</td>
      <td>5</td>
      <td>0.00</td>
      <td>170.43</td>
      <td>-11.29</td>
      <td>-22.21</td>
      <td>117.87</td>
      <td>1980-01-05</td>
      <td>01013500</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7977415</th>
      <td>2014</td>
      <td>12</td>
      <td>27</td>
      <td>0.00</td>
      <td>196.83</td>
      <td>9.55</td>
      <td>1.93</td>
      <td>704.03</td>
      <td>2014-12-27</td>
      <td>14400000</td>
      <td>3800.0</td>
    </tr>
    <tr>
      <th>7977416</th>
      <td>2014</td>
      <td>12</td>
      <td>28</td>
      <td>4.93</td>
      <td>133.57</td>
      <td>7.90</td>
      <td>1.59</td>
      <td>687.43</td>
      <td>2014-12-28</td>
      <td>14400000</td>
      <td>3220.0</td>
    </tr>
    <tr>
      <th>7977417</th>
      <td>2014</td>
      <td>12</td>
      <td>29</td>
      <td>6.54</td>
      <td>134.34</td>
      <td>5.51</td>
      <td>-0.78</td>
      <td>581.32</td>
      <td>2014-12-29</td>
      <td>14400000</td>
      <td>2810.0</td>
    </tr>
    <tr>
      <th>7977418</th>
      <td>2014</td>
      <td>12</td>
      <td>30</td>
      <td>4.60</td>
      <td>149.87</td>
      <td>4.78</td>
      <td>-2.52</td>
      <td>512.63</td>
      <td>2014-12-30</td>
      <td>14400000</td>
      <td>2460.0</td>
    </tr>
    <tr>
      <th>7977419</th>
      <td>2014</td>
      <td>12</td>
      <td>31</td>
      <td>0.00</td>
      <td>223.80</td>
      <td>9.78</td>
      <td>-0.89</td>
      <td>575.12</td>
      <td>2014-12-31</td>
      <td>14400000</td>
      <td>2160.0</td>
    </tr>
  </tbody>
</table>
<p>7977420 rows × 11 columns</p>
</div>



check the missing values


```python
print("check the varying features")
print(varying_features.isnull().sum())
print("____________________________")
print("check the static features")
print(static_features.isnull().sum())
```

    check the varying features
    Year            0
    Mnth            0
    Day             0
    prcp(mm/day)    0
    srad(W/m2)      0
    tmax(C)         0
    tmin(C)         0
    vp(Pa)          0
    datetime        0
    gauge_id        0
    runoff          0
    dtype: int64
    ____________________________
    check the static features
    gauge_id                0
    elev_mean               0
    slope_mean              0
    area_gages2             0
    p_mean                  0
    pet_mean                0
    aridity                 0
    p_seasonality           0
    frac_snow               0
    high_prec_freq          0
    high_prec_dur           0
    high_prec_timing        0
    low_prec_freq           0
    low_prec_dur            0
    frac_forest             0
    lai_max                 0
    lai_diff                0
    gvf_max                 0
    gvf_diff                0
    soil_depth_pelletier    0
    soil_depth_statsgo      0
    soil_porosity           0
    soil_conductivity       0
    max_water_content       0
    sand_frac               0
    silt_frac               0
    clay_frac               0
    carbonate_rocks_frac    0
    geol_permeability       0
    dtype: int64
    


```python
guage_id = "01013500" #choose a basin to test
axes = varying_features[varying_features['gauge_id']==guage_id].plot(
    x="datetime",
    y=["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)", "runoff"] if rainfall_source == "daymet" else ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)", "runoff"],
    subplots=True,
    figsize=(20, 15),
)
for ax in axes:
    ax.legend(loc='upper left')
    ax.autoscale(axis='x', tight=True)
```

___
<h3 style='pointer-events: none;'>2.Varying features preprocessing</h3>

describe the varying features


```python
# drop Year Mnth Day
varying_features.drop(
    columns=["gauge_id", "Year", "Mnth", "Day"]
).describe()
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
      <th>prcp(mm/day)</th>
      <th>srad(W/m2)</th>
      <th>tmax(C)</th>
      <th>tmin(C)</th>
      <th>vp(Pa)</th>
      <th>datetime</th>
      <th>runoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.977420e+06</td>
      <td>7.977420e+06</td>
      <td>7.977420e+06</td>
      <td>7.977420e+06</td>
      <td>7.977420e+06</td>
      <td>7977420</td>
      <td>7.977420e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.235265e+00</td>
      <td>3.403719e+02</td>
      <td>1.658163e+01</td>
      <td>4.178058e+00</td>
      <td>9.507575e+02</td>
      <td>1997-09-11 12:32:49.701484032</td>
      <td>3.183479e+02</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>1.305000e+01</td>
      <td>-3.807000e+01</td>
      <td>-4.884000e+01</td>
      <td>0.000000e+00</td>
      <td>1980-01-01 00:00:00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>2.418000e+02</td>
      <td>8.170000e+00</td>
      <td>-2.500000e+00</td>
      <td>4.640500e+02</td>
      <td>1989-02-25 00:00:00</td>
      <td>8.800000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
      <td>3.423200e+02</td>
      <td>1.767000e+01</td>
      <td>4.110000e+00</td>
      <td>7.415400e+02</td>
      <td>1997-09-28 00:00:00</td>
      <td>4.900000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.810000e+00</td>
      <td>4.384500e+02</td>
      <td>2.557000e+01</td>
      <td>1.179000e+01</td>
      <td>1.311480e+03</td>
      <td>2006-04-30 00:00:00</td>
      <td>2.000000e+02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000e+02</td>
      <td>8.000000e+02</td>
      <td>4.550000e+01</td>
      <td>2.850000e+01</td>
      <td>3.659250e+03</td>
      <td>2014-12-31 00:00:00</td>
      <td>1.240000e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.736499e+00</td>
      <td>1.315360e+02</td>
      <td>1.116497e+01</td>
      <td>1.016901e+01</td>
      <td>6.527316e+02</td>
      <td>NaN</td>
      <td>1.206547e+03</td>
    </tr>
  </tbody>
</table>
</div>



visualize the distribution of the varying features


```python
if rainfall_source == "daymet":
    varying_features_str = ["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)", "runoff"]
else:
    varying_features_str = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)", "runoff"]

for i, col in enumerate(varying_features[varying_features_str]):
    plt.subplot(2, 3, i+1)
    if col in ["prcp(mm/day)", "runoff"]:
        plt.yscale("log")
    varying_features[col].hist(bins=50, figsize=(15, 7.5)).autoscale(axis='x', tight=True)
    plt.title(f"Distribution of {col}")
# make sure the subplots are not overlapping
plt.tight_layout()
```

Clearly, srad, tmax, tmin, and vp are normally distributed, while prcp and runoff are not.
So we use log scale for prcp and runoff.


```python
# log scale for prcp
if rainfall_source == "daymet":
    varying_features["prcp(mm/day)"] = varying_features["prcp(mm/day)"].apply(lambda x: np.log(x+1))
else:
    varying_features["PRCP(mm/day)"] = varying_features["PRCP(mm/day)"].apply(lambda x: np.log(x+1))
# log scale for runoff
varying_features["runoff"] = varying_features["runoff"].apply(lambda x: np.log(x+1))
```

Year maybe a useful feature, which can be used to capture the trend of the time series.
We subtract the first year from the Year column.


```python
varying_features["Year"] = varying_features["Year"] - varying_features["Year"].min()
```

Mnth and Day represent the periodicity of the time series..
We use sin and cos to encode the periodicity.


```python
varying_features["Mnth_sin"] = np.sin(2 * np.pi * varying_features["Mnth"] / 12)
varying_features["Mnth_cos"] = np.cos(2 * np.pi * varying_features["Mnth"] / 12)
# consider the true day of the month
varying_features["Day_sin"] = np.sin(2 * np.pi * varying_features["Day"] / 
                                        varying_features["datetime"].dt.daysinmonth)
varying_features["Day_cos"] = np.cos(2 * np.pi * varying_features["Day"] /
                                        varying_features["datetime"].dt.daysinmonth) 
```



drop the original Mnth and Day


```python
varying_features.drop(columns=["Mnth", "Day"], inplace=True)
varying_features
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
      <th>prcp(mm/day)</th>
      <th>srad(W/m2)</th>
      <th>tmax(C)</th>
      <th>tmin(C)</th>
      <th>vp(Pa)</th>
      <th>datetime</th>
      <th>gauge_id</th>
      <th>runoff</th>
      <th>Mnth_sin</th>
      <th>Mnth_cos</th>
      <th>Day_sin</th>
      <th>Day_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>153.40</td>
      <td>-6.54</td>
      <td>-16.30</td>
      <td>171.69</td>
      <td>1980-01-01</td>
      <td>01013500</td>
      <td>6.486161</td>
      <td>5.000000e-01</td>
      <td>0.866025</td>
      <td>2.012985e-01</td>
      <td>0.979530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.000000</td>
      <td>145.27</td>
      <td>-6.18</td>
      <td>-15.22</td>
      <td>185.94</td>
      <td>1980-01-02</td>
      <td>01013500</td>
      <td>6.463029</td>
      <td>5.000000e-01</td>
      <td>0.866025</td>
      <td>3.943559e-01</td>
      <td>0.918958</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.000000</td>
      <td>146.96</td>
      <td>-9.89</td>
      <td>-18.86</td>
      <td>138.39</td>
      <td>1980-01-03</td>
      <td>01013500</td>
      <td>6.439350</td>
      <td>5.000000e-01</td>
      <td>0.866025</td>
      <td>5.712682e-01</td>
      <td>0.820763</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.000000</td>
      <td>146.20</td>
      <td>-10.98</td>
      <td>-19.76</td>
      <td>120.06</td>
      <td>1980-01-04</td>
      <td>01013500</td>
      <td>6.431331</td>
      <td>5.000000e-01</td>
      <td>0.866025</td>
      <td>7.247928e-01</td>
      <td>0.688967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.000000</td>
      <td>170.43</td>
      <td>-11.29</td>
      <td>-22.21</td>
      <td>117.87</td>
      <td>1980-01-05</td>
      <td>01013500</td>
      <td>6.406880</td>
      <td>5.000000e-01</td>
      <td>0.866025</td>
      <td>8.486443e-01</td>
      <td>0.528964</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7977415</th>
      <td>34</td>
      <td>0.000000</td>
      <td>196.83</td>
      <td>9.55</td>
      <td>1.93</td>
      <td>704.03</td>
      <td>2014-12-27</td>
      <td>14400000</td>
      <td>8.243019</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>-7.247928e-01</td>
      <td>0.688967</td>
    </tr>
    <tr>
      <th>7977416</th>
      <td>34</td>
      <td>1.780024</td>
      <td>133.57</td>
      <td>7.90</td>
      <td>1.59</td>
      <td>687.43</td>
      <td>2014-12-28</td>
      <td>14400000</td>
      <td>8.077447</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>-5.712682e-01</td>
      <td>0.820763</td>
    </tr>
    <tr>
      <th>7977417</th>
      <td>34</td>
      <td>2.020222</td>
      <td>134.34</td>
      <td>5.51</td>
      <td>-0.78</td>
      <td>581.32</td>
      <td>2014-12-29</td>
      <td>14400000</td>
      <td>7.941296</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>-3.943559e-01</td>
      <td>0.918958</td>
    </tr>
    <tr>
      <th>7977418</th>
      <td>34</td>
      <td>1.722767</td>
      <td>149.87</td>
      <td>4.78</td>
      <td>-2.52</td>
      <td>512.63</td>
      <td>2014-12-30</td>
      <td>14400000</td>
      <td>7.808323</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>-2.012985e-01</td>
      <td>0.979530</td>
    </tr>
    <tr>
      <th>7977419</th>
      <td>34</td>
      <td>0.000000</td>
      <td>223.80</td>
      <td>9.78</td>
      <td>-0.89</td>
      <td>575.12</td>
      <td>2014-12-31</td>
      <td>14400000</td>
      <td>7.678326</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
      <td>-2.449294e-16</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>7977420 rows × 13 columns</p>
</div>



scale the varying features except Year, Mnth_sin, Mnth_cos, Day_sin, Day_cos


```python
varying_features_scaler = MinMaxScaler()
varying_features[varying_features_str] = varying_features_scaler.fit_transform(
    varying_features[varying_features_str]
)
# save the scaler and varying features
with open(f"../data/varying_features_{rainfall_source}_MinMaxScaler.pickle", "wb") as f:
    pickle.dump(varying_features_scaler, f)
varying_features.to_csv(f"../data/varying_features_{rainfall_source}_preprocessed.csv", index=False)
```

___
<h3 style='pointer-events: none;'>3. Static features preprocessing</h3>

```python
static_features
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
      <th>gauge_id</th>
      <th>elev_mean</th>
      <th>slope_mean</th>
      <th>area_gages2</th>
      <th>p_mean</th>
      <th>pet_mean</th>
      <th>aridity</th>
      <th>p_seasonality</th>
      <th>frac_snow</th>
      <th>high_prec_freq</th>
      <th>...</th>
      <th>soil_depth_pelletier</th>
      <th>soil_depth_statsgo</th>
      <th>soil_porosity</th>
      <th>soil_conductivity</th>
      <th>max_water_content</th>
      <th>sand_frac</th>
      <th>silt_frac</th>
      <th>clay_frac</th>
      <th>carbonate_rocks_frac</th>
      <th>geol_permeability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01013500</td>
      <td>250.31</td>
      <td>21.64152</td>
      <td>2252.70</td>
      <td>3.126679</td>
      <td>1.971555</td>
      <td>0.630559</td>
      <td>0.187940</td>
      <td>0.313440</td>
      <td>12.95</td>
      <td>...</td>
      <td>7.404762</td>
      <td>1.248408</td>
      <td>0.461149</td>
      <td>1.106522</td>
      <td>0.558055</td>
      <td>27.841827</td>
      <td>55.156940</td>
      <td>16.275732</td>
      <td>0.000000</td>
      <td>-14.7019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01022500</td>
      <td>92.68</td>
      <td>17.79072</td>
      <td>573.60</td>
      <td>3.608126</td>
      <td>2.119256</td>
      <td>0.587356</td>
      <td>-0.114530</td>
      <td>0.245259</td>
      <td>20.55</td>
      <td>...</td>
      <td>17.412808</td>
      <td>1.491846</td>
      <td>0.415905</td>
      <td>2.375005</td>
      <td>0.626229</td>
      <td>59.390156</td>
      <td>28.080937</td>
      <td>12.037646</td>
      <td>0.000000</td>
      <td>-14.2138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01030500</td>
      <td>143.80</td>
      <td>12.79195</td>
      <td>3676.17</td>
      <td>3.274405</td>
      <td>2.043594</td>
      <td>0.624111</td>
      <td>0.047358</td>
      <td>0.277018</td>
      <td>17.15</td>
      <td>...</td>
      <td>19.011414</td>
      <td>1.461363</td>
      <td>0.459091</td>
      <td>1.289807</td>
      <td>0.653020</td>
      <td>32.235458</td>
      <td>51.779182</td>
      <td>14.776824</td>
      <td>0.052140</td>
      <td>-14.4918</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01031500</td>
      <td>247.80</td>
      <td>29.56035</td>
      <td>769.05</td>
      <td>3.522957</td>
      <td>2.071324</td>
      <td>0.587950</td>
      <td>0.104091</td>
      <td>0.291836</td>
      <td>18.90</td>
      <td>...</td>
      <td>7.252557</td>
      <td>1.279047</td>
      <td>0.450236</td>
      <td>1.373292</td>
      <td>0.559123</td>
      <td>35.269030</td>
      <td>50.841232</td>
      <td>12.654125</td>
      <td>0.026258</td>
      <td>-14.8410</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01047000</td>
      <td>310.38</td>
      <td>49.92122</td>
      <td>909.10</td>
      <td>3.323146</td>
      <td>2.090024</td>
      <td>0.628929</td>
      <td>0.147776</td>
      <td>0.280118</td>
      <td>20.10</td>
      <td>...</td>
      <td>5.359655</td>
      <td>1.392779</td>
      <td>0.422749</td>
      <td>2.615154</td>
      <td>0.561181</td>
      <td>55.163133</td>
      <td>34.185443</td>
      <td>10.303622</td>
      <td>0.000000</td>
      <td>-14.4819</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>631</th>
      <td>14309500</td>
      <td>709.83</td>
      <td>110.42527</td>
      <td>224.92</td>
      <td>4.977781</td>
      <td>3.122204</td>
      <td>0.627228</td>
      <td>-0.995847</td>
      <td>0.061255</td>
      <td>15.10</td>
      <td>...</td>
      <td>0.894595</td>
      <td>0.894531</td>
      <td>0.442804</td>
      <td>1.335304</td>
      <td>0.395703</td>
      <td>37.751217</td>
      <td>38.879406</td>
      <td>23.213862</td>
      <td>0.000000</td>
      <td>-14.8976</td>
    </tr>
    <tr>
      <th>632</th>
      <td>14316700</td>
      <td>952.26</td>
      <td>119.08920</td>
      <td>587.90</td>
      <td>4.543400</td>
      <td>2.277630</td>
      <td>0.501305</td>
      <td>-0.821172</td>
      <td>0.176337</td>
      <td>14.75</td>
      <td>...</td>
      <td>0.879292</td>
      <td>1.340004</td>
      <td>0.443107</td>
      <td>1.288301</td>
      <td>0.584644</td>
      <td>37.238495</td>
      <td>38.519396</td>
      <td>24.363634</td>
      <td>0.000000</td>
      <td>-13.5958</td>
    </tr>
    <tr>
      <th>633</th>
      <td>14325000</td>
      <td>656.53</td>
      <td>124.96889</td>
      <td>443.07</td>
      <td>6.297437</td>
      <td>2.434652</td>
      <td>0.386610</td>
      <td>-0.952055</td>
      <td>0.030203</td>
      <td>14.60</td>
      <td>...</td>
      <td>0.990318</td>
      <td>0.892189</td>
      <td>0.442249</td>
      <td>1.425770</td>
      <td>0.388650</td>
      <td>38.961578</td>
      <td>40.860260</td>
      <td>20.068726</td>
      <td>0.000000</td>
      <td>-15.1799</td>
    </tr>
    <tr>
      <th>634</th>
      <td>14362250</td>
      <td>875.67</td>
      <td>109.93127</td>
      <td>41.42</td>
      <td>2.781676</td>
      <td>3.325188</td>
      <td>1.195390</td>
      <td>-0.985486</td>
      <td>0.141500</td>
      <td>20.45</td>
      <td>...</td>
      <td>0.625000</td>
      <td>0.800111</td>
      <td>0.442872</td>
      <td>1.363910</td>
      <td>0.348779</td>
      <td>37.914394</td>
      <td>39.602460</td>
      <td>22.404372</td>
      <td>0.000000</td>
      <td>-12.5264</td>
    </tr>
    <tr>
      <th>635</th>
      <td>14400000</td>
      <td>625.31</td>
      <td>98.81802</td>
      <td>702.63</td>
      <td>5.556071</td>
      <td>2.279668</td>
      <td>0.410302</td>
      <td>-1.015946</td>
      <td>0.024330</td>
      <td>19.30</td>
      <td>...</td>
      <td>0.929015</td>
      <td>1.071426</td>
      <td>0.452074</td>
      <td>0.941310</td>
      <td>0.494187</td>
      <td>30.026587</td>
      <td>42.214515</td>
      <td>27.805122</td>
      <td>0.000000</td>
      <td>-14.7163</td>
    </tr>
  </tbody>
</table>
<p>636 rows × 29 columns</p>
</div>




scale the static features


```python
static_features_scaler = MinMaxScaler()
static_features_str = pd.read_excel("../data/static features.xlsx")
print(static_features_str)
static_features_str = static_features_str["features"].tolist()
static_features_str.remove("high_prec_timing")
```

                    features                                        description
    0              elev_mean                           catchment mean elevation
    1             slope_mean                               catchment mean slope
    2            area_gages2                  catchment area (GAGESII estimate)
    3                 p_mean                           mean daily precipitation
    4               pet_mean  mean daily PET [estimated by N15 using Priestl...
    5                aridity  aridity (PET/P, ratio of mean PET [estimated b...
    6          p_seasonality  seasonality and timing of precipitation (estim...
    7              frac_snow  fraction of precipitation falling as snow (i.e...
    8         high_prec_freq  frequency of high precipitation days ( >= 5 ti...
    9          high_prec_dur  average duration of high precipitation events ...
    10      high_prec_timing  season during which most high precipitation da...
    11         low_prec_freq                 frequency of dry days ( <1 mm/day)
    12          low_prec_dur  average duration of dry periods (number of con...
    13           frac_forest                                    forest fraction
    14               lai_max  maximum monthly mean of the leaf area index (b...
    15              lai_diff  difference between the maximum and mimumum mon...
    16               gvf_max  maximum monthly mean of the green vegetation f...
    17              gvf_diff  difference between the maximum and mimumum mon...
    18  soil_depth_pelletier                     depth to bedrock (maximum 50m)
    19    soil_depth_statsgo  soil depth (maximum 1.5m, layers marked as wat...
    20         soil_porosity  volumetric porosity (saturated volumetric wate...
    21     soil_conductivity  saturated hydraulic conductivity (estimated us...
    22     max_water_content  maximum water content (combination of porosity...
    23             sand_frac  sand fraction (of the soil material smaller th...
    24             silt_frac  silt fraction (of the soil material smaller th...
    25             clay_frac  clay fraction (of the soil material smaller th...
    26  carbonate_rocks_frac  fraction of the catchment area characterized a...
    27     geol_permeability                    subsurface permeability (log10)
    


```python
static_features[static_features_str] = static_features_scaler.fit_transform(
    static_features[static_features_str]
)
# save the scaler and static features
with open("../data/static_features_MinMaxScaler.pickle", "wb") as f:
    pickle.dump(static_features_scaler, f)
static_features.to_csv("../data/static_features_preprocessed.csv", index=False)
```

Load pickled scaler after modeling


```python
# with open("../data/varying_features_MinMaxScaler.pickle", "rb") as f:
#     varying_features_scaler = pickle.load(f)
# with open("../data/static_features_MinMaxScaler.pickle", "rb") as f:
#     static_features_scaler = pickle.load(f)
# varying_features = varying_features_scaler.transform(varying_features[["prcp(mm/day)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)", "runoff"]])
# static_features = static_features_scaler.transform(static_features[static_features_str])
```
