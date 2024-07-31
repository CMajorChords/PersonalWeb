<h2 style='pointer-events: none;'>Xinanjiang Model</h2>

The details of each function can be found in the progress of 2023-10-12
<h3 style='pointer-events: none;'>1 Preparation</h3>
<h4 style='pointer-events: none;'>Import data</h4>

```
import numpy as np
import pandas as pd
data = pd.read_excel('Data/data.xlsx')
data_length = len(data)
precipitation = data['precipitation'].values
evaporation_potential = data['evaporation_potential'].values
runoff_observed = data['runoff_observed'].values
```
<h3 style='pointer-events: none;'>2 runoff yield model</h3>
<h4 style='pointer-events: none;'>Boundary conditions</h4>

ModelConceptional.xaj_function is a function library, which can be found in the progress of 2023-10-12
```
from ModelConceptional.xaj_function import evaporate_3layer, runoff_generate_stored_full, water_content_update
evaporation_upper = np.zeros(data_length)
evaporation_lower = np.zeros(data_length)
evaporation_deeper = np.zeros(data_length)
evaporation = np.zeros(data_length)
water_content_upper = np.zeros(data_length)
water_content_lower = np.zeros(data_length)
water_content_deeper = np.zeros(data_length)
water_content = np.zeros(data_length)
runoff = np.zeros(data_length)
net_rainfall = np.zeros(data_length)
water_content_upper[0] = 0
water_content_lower[0] = 30
water_content_deeper[0] = 40
water_content[0] = water_content_upper[0] + water_content_lower[0] + water_content_deeper[0]
```
<h4 style='pointer-events: none;'>runoff yield model parameters</h4>

```
water_content_upper_max = 20
water_content_lower_max = 60
water_content_deeper_max = 40
water_content_max = 120
B = 0.3
water_content_max_curve_max = water_content_max*(1+B)
C = 0.18
IM = 0.01
```
<h4 style='pointer-events: none;'>runoff yield model calculation</h4>

```
for i in range(1, data_length):
    # 更新土壤含水量
    (water_content_upper[i],
     water_content_lower[i],
     water_content_deeper[i],
     water_content[i]
     ) = water_content_update(
        water_content_upper=water_content_upper[i - 1],
        water_content_lower=water_content_lower[i - 1],
        water_content_deeper=water_content_deeper[i - 1],
        evaporation_upper=evaporation_upper[i - 1],
        evaporation_lower=evaporation_lower[i - 1],
        evaporation_deeper=evaporation_deeper[i - 1],
        precipitation=precipitation[i - 1],
        runoff=runoff[i - 1],
        water_content_upper_max=water_content_upper_max,
        water_content_lower_max=water_content_lower_max,
        water_content_deeper_max=water_content_deeper_max,
    )
    # 计算蒸发量
    (evaporation_upper[i],
     evaporation_lower[i],
     evaporation_deeper[i],
     evaporation[i]
     ) = evaporate_3layer(
        water_content_upper=water_content_upper[i],
        water_content_lower=water_content_lower[i],
        precipitation=precipitation[i],
        evaporation_potential=evaporation_potential[i],
        evaporation_coefficient=C,
        water_content_lower_max=water_content_lower_max,
    )
    # 蓄满产流模型
    (runoff[i],
     net_rainfall[i]
     ) = runoff_generate_stored_full(
        water_content=water_content[i-1],
        precipitation=precipitation[i],
        evaporation=evaporation[i],
        water_content_max_curve_max=water_content_max_curve_max,
        impermeable_area_ratio=IM,
        water_content_max=water_content_max,
        b=B,
    )
```
<h3 style='pointer-events: none;'>3 confluence model</h3>
<h4 style='pointer-events: none;'>Boundary conditions</h4>

```
from ModelConceptional.xaj_function import water_division_3, linear_reservoir
runoff_surface = np.zeros(data_length)
runoff_inter = np.zeros(data_length)
runoff_ground = np.zeros(data_length)
free_water_content = np.zeros(data_length)
runoff_generate_area_ratio = np.zeros(data_length)
flow_surface = np.zeros(data_length)
flow_inter = np.zeros(data_length)
flow_ground = np.zeros(data_length)
flow_total = np.zeros(data_length)
flow = np.zeros(data_length)
flow[0] = 10
flow_surface[0] = 10
flow_inter[0] = 10
flow_ground[0] = 10
flow_total[0] = flow_surface[0] + flow_inter[0] + flow_ground[0]
runoff_generate_area_ratio[0] = 0.1
```
<h4 style='pointer-events: none;'>confluence model parameters</h4>

```
free_water_content_max = 20
outflow_coefficient_inter = 0.6
outflow_coefficient_ground = 0.13
depletion_constant_surface = 0.09
depletion_constant_inter = 0.7
EX = 1.5
depletion_constant_ground = 0.98
F = 290.1 # 单位：km²
time_lag = 24 # 单位：h
U = F / (3.6 * time_lag)
```
<h4 style='pointer-events: none;'>confluence model calculation</h4>

```
for i in range(1, data_length):
    # 如果净雨量为负，产流面积比例为0
    if net_rainfall[i] <= 0:
        runoff_generate_area_ratio[i] = 0
    # 如果净雨量为正，产流面积比例为产流/净雨量
    else:
        runoff_generate_area_ratio[i] = runoff[i]/net_rainfall[i]
    # 分水源计算
    (runoff_surface[i],
     runoff_inter[i],
     runoff_ground[i],
     free_water_content[i]) = water_division_3(
        free_water_content_last=free_water_content[i - 1],
        runoff_generate_area_ratio_last=runoff_generate_area_ratio[i - 1],
        net_rainfall=net_rainfall[i],
        runoff=runoff[i],
        ex=EX,
        free_water_content_max=free_water_content_max,
        outflow_coefficient_inter=outflow_coefficient_inter,
        outflow_coefficient_ground=outflow_coefficient_ground,
    )
    # 线性水库汇流
    flow_surface[i] = linear_reservoir(
        runoff_component=runoff_surface[i],
        flow_last=flow_surface[i - 1],
        runoff_generate_area_ratio_last=runoff_generate_area_ratio[i],
        depletion_constant=depletion_constant_surface,
        unit_from_depth_2_volume=U,
    )
    flow_inter[i] = linear_reservoir(
        runoff_component=runoff_inter[i],
        flow_last=flow_inter[i - 1],
        runoff_generate_area_ratio_last=runoff_generate_area_ratio[i],
        depletion_constant=depletion_constant_inter,
        unit_from_depth_2_volume=U
    )
    flow_ground[i] = linear_reservoir(
        runoff_component=runoff_ground[i],
        flow_last=flow_ground[i - 1],
        runoff_generate_area_ratio_last=runoff_generate_area_ratio[i],
        depletion_constant=depletion_constant_ground,
        unit_from_depth_2_volume=U
    )
    flow_total[i] = flow_surface[i] + flow_inter[i] + flow_ground[i]
    flow[i] = linear_reservoir(
        flow_last=flow[i - 1],
        runoff_component=flow_total[i - 1],
        depletion_constant=depletion_constant_surface,
        runoff_generate_area_ratio_last=1,
        unit_from_depth_2_volume=1
    )
```