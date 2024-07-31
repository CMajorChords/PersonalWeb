<h2 style='pointer-events: none;'>Darts with CAMELS</h2>

本内容有很多图片，建议在[我的colab](https://drive.google.com/file/d/16LnH12fjejtbzzoIDNYyucO0HlrTbtRV/view?usp=sharing)(需要科学上网)中查看，效果更好。
Dart 是一个用于轻松操作和预测时间序列的  库。它包含各种模型，从经典的 ARIMA 到深度神经网络。Darts 也很容易扩展，可以快速添加具有附加模型和转换器的新模块。
下面的笔记本显示了一个典型的工作流，包括使用 CAMELS 数据集加载数据、数据清理、模型训练和预测。  
CAMELS 是一个大型的降雨径流数据集，包含美国 671 个集水区和 22 个水文变量，是水文建模的流行基准数据集。这个数据用来验证模型的好坏还蛮好用的。
```
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.dataframe = pd.DataFrame
from darts import TimeSeries
```
<h3 style='pointer-events: none;'>1. 把 CAMELS 数据集转换成 TimeSeries</h3>

TimeSeries 是 Darts 中的核心对象。它是一个带有时间索引的 pandas Series，可以包含任何类型的数据，例如浮点数、整数或字符串。
<h4 style='pointer-events: none;'>1.1 首先使用 CAMELS 类加载 CAMELS 数据集：</h4>

```
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
<h4 style='pointer-events: none;'>1.2 然后，我们可以使用 TimeSeries.from_dataframe() 方法将数据转换为 TimeSeries 对象：</h4>

```
basin_data = TimeSeries.from_dataframe(basin["data"].drop(columns=["Year", "Mnth", "Day"]))
basin_data[pd.Timestamp("1990-01-01"):pd.Timestamp("1990-12-31")].plot()
runoff = basin_data["runoff"]
```

    
<h5 style='pointer-events: none;'>划分</h5>

```
runoff1, runoff2 = runoff.split_after(pd.Timestamp("2000-01-01"))
runoff1.plot(label="runoff before 2000-01-01")
runoff2.plot(label="runoff after 2000-01-01")
```





    
<h5 style='pointer-events: none;'>切片</h5>

```
runoff1 = runoff[pd.Timestamp("1990-01-01"):pd.Timestamp("1990-12-31")]
runoff2 = runoff[pd.Timestamp("1991-01-01"):pd.Timestamp("1991-12-31")]
runoff1.plot(label="runoff before 2000-01-01")
runoff2.plot(label="runoff after 2000-01-01")
```




    
<h5 style='pointer-events: none;'>算术运算</h5>

```
runoff_noise = TimeSeries.from_times_and_values(
    runoff.time_index, np.random.randn(len(runoff))
)
runoff.plot(label="runoff")
(runoff / 2 + 20 * runoff_noise).plot(label="runoff / 2 + 20 * noise")
```





    
<h5 style='pointer-events: none;'>堆叠</h5>

Concatenating a new dimension to produce a new single multivariate time series.


```
(runoff / 1000).stack(runoff_noise).plot()
```





    
<h5 style='pointer-events: none;'>mapping操作</h5>

```
runoff.map(lambda x: x ** 2).plot(label="runoff squared")
```





    
<h5 style='pointer-events: none;'>添加一些日期时间属性作为额外的维度（产生多变量序列）</h5>

```
(runoff / 1000 ).add_datetime_attribute("month").plot()
```





    
<h5 style='pointer-events: none;'>添加一些假期作为额外的维度（产生多变量序列）</h5>

```
(runoff / 5000).add_holidays("CN").plot()
```





    
<h5 style='pointer-events: none;'>差分</h5>

```
runoff.diff().plot()
```





    
<h3 style='pointer-events: none;'>2. 创建训练集和验证集</h3>

```
train, val = runoff.split_before(pd.Timestamp("2000-01-01"))
train.plot(label="train")
val.plot(label="validation")
```





    
<h3 style='pointer-events: none;'>3. 模型训练和预测</h3>
<h4 style='pointer-events: none;'>3.1 玩具模型</h4>

```
from darts.models import NaiveSeasonal

naive_model = NaiveSeasonal(K=1)
naive_model.fit(train)
naive_forecast = naive_model.predict(len(val))

runoff.plot(label="actual")
naive_forecast.plot(label="forecast")
```

    D:\Software\miniconda3\Lib\site-packages\statsforecast\core.py:25: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from tqdm.autonotebook import tqdm
    





    
<h4 style='pointer-events: none;'>3.2 检查季节性</h4>

```
from darts.utils.statistics import check_seasonality, plot_acf
plot_acf(runoff, alpha=0.05)
```


    

    


检查季节时间步


```
for m in range(2, 25):
    is_seasonal, period = check_seasonality(train, m=m, alpha=0.05)
    if is_seasonal:
        print("There is seasonality of order {}.".format(period))
# 没有季节性好像
```
