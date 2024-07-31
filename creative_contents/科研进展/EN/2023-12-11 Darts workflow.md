<h2 style='pointer-events: none;'>Darts workflow with CAMELS</h2>

There are many graphs in this notebook, so it is recommended to run it on my [colab](https://drive.google.com/file/d/16LnH12fjejtbzzoIDNYyucO0HlrTbtRV/view?usp=sharing).
Darts is a Python library for easy manipulation and forecasting of time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. Darts is also very easy to extend, and new modules with additional models and transformers can be quickly added.
The notebook below shows a typical workflow including data loading with CAMELS dataset, data cleaning, model training, and forecasting.
CAMELS is a large rainfall-runoff dataset for the continental United States. It contains 671 catchments and 22 hydrologic variables, and is a popular benchmark dataset for hydrologic modeling.



```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.dataframe = pd.DataFrame
from darts import TimeSeries
```
<h3 style='pointer-events: none;'>1. TimeSeries of CAMELS data</h3>

TimeSeries is the core object in Darts. It is a pandas Series with a time index, and can contain any type of data, such as floats, integers, or strings.
<h4 style='pointer-events: none;'>1.1 First we load the CAMELS data using the CAMELS class：</h4>

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

    def get_basin_data(self, gauge_id, rainfall_data_source="daymet") -> dict:
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
        rainfall_data.drop(columns=["Hr", "dayl(s)"], axis=1, inplace=True)
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
        return basin

camels = CAMELS("../data")
basin = camels.get_basin_data("01013500")
```
<h4 style='pointer-events: none;'>1.2 We can now create a TimeSeries from the basin data:</h4>

```python
basin_data = TimeSeries.from_dataframe(basin["data"].drop(columns=["Year", "Mnth", "Day"]))
basin_data[pd.Timestamp("1990-01-01"):pd.Timestamp("1990-12-31")].plot()
runoff = basin_data["runoff"]
```

    
<h5 style='pointer-events: none;'>splitting</h5>

```python
runoff1, runoff2 = runoff.split_after(pd.Timestamp("2000-01-01"))
runoff1.plot(label="runoff before 2000-01-01")
runoff2.plot(label="runoff after 2000-01-01")
```





    
<h5 style='pointer-events: none;'>slicing</h5>

```python
runoff1 = runoff[pd.Timestamp("1990-01-01"):pd.Timestamp("1990-12-31")]
runoff2 = runoff[pd.Timestamp("1991-01-01"):pd.Timestamp("1991-12-31")]
runoff1.plot(label="runoff before 2000-01-01")
runoff2.plot(label="runoff after 2000-01-01")
```




    
<h5 style='pointer-events: none;'>arithmetic operations</h5>

```python
runoff_noise = TimeSeries.from_times_and_values(
    runoff.time_index, np.random.randn(len(runoff))
)
runoff.plot(label="runoff")
(runoff / 2 + 20 * runoff_noise).plot(label="runoff / 2 + 20 * noise")
```





    
<h5 style='pointer-events: none;'>stacking</h5>

Concatenating a new dimension to produce a new single multivariate time series.


```python
(runoff / 1000).stack(runoff_noise).plot()
```





    
<h5 style='pointer-events: none;'>mapping</h5>

```python
runoff.map(lambda x: x ** 2).plot(label="runoff squared")
```





    
<h5 style='pointer-events: none;'>Adding some datetime attribute as an extra dimension (yielding a multivariate series):</h5>

```python
(runoff / 1000 ).add_datetime_attribute("month").plot()
```





    
<h5 style='pointer-events: none;'>Adding some datetime attribute as an extra dimension (yielding a multivariate series):</h5>

```python
(runoff / 5000).add_holidays("CN").plot()
```





    
<h5 style='pointer-events: none;'>differencing</h5>

```python
runoff.diff().plot()
```





    
<h3 style='pointer-events: none;'>2. Creating a training and validation set</h3>

```python
train, val = runoff.split_before(pd.Timestamp("2000-01-01"))
train.plot(label="train")
val.plot(label="validation")
```





    
<h3 style='pointer-events: none;'>3. Training forecasting models and making predictions</h3>
<h4 style='pointer-events: none;'>3.1 Playing with toy models</h4>

```python
from darts.models import NaiveSeasonal

naive_model = NaiveSeasonal(K=1)
naive_model.fit(train)
naive_forecast = naive_model.predict(len(val))

runoff.plot(label="actual")
naive_forecast.plot(label="forecast")
```

    D:\Software\miniconda3\Lib\site-packages\statsforecast\core.py:25: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from tqdm.autonotebook import tqdm
    





    
<h4 style='pointer-events: none;'>3.2 Checking for seasonality</h4>

```python
from darts.utils.statistics import check_seasonality, plot_acf
plot_acf(runoff, alpha=0.05)
```


    

    


check how many lags are needed to capture the seasonality


```python
for m in range(2, 25):
    is_seasonal, period = check_seasonality(train, m=m, alpha=0.05)
    if is_seasonal:
        print("There is seasonality of order {}.".format(period))
# Oops, no seasonality detected
```
