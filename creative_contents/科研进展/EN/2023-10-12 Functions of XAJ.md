<h2 style='pointer-events: none;'>Functions of Three Water Source XAJ Model</h2>
<h4 style='pointer-events: none;'> 1 Three-layer Evaporation Model</h4>

- 当超过第一临界土壤含水量时：
        蒸发主要发生在土壤上层，上层土壤蒸发量即为潜在蒸发量，该阶段供水充分；
- 当在第二和第一临界土壤含水量之间，扣除表层蒸发的潜在蒸发量比下层土壤能蒸发的少：
        上层土壤含水量和降水量全部蒸发，下层蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为下层土壤含水比例；
- 当在第二和第一临界土壤含水量之间，扣除表层蒸发的潜在蒸发量比下层土壤能蒸发的多：
        上层土壤含水量和降水量全部蒸发，下层蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为蒸发扩散系数；
- 当不超过第二临界土壤含水量时：
        降水量、上层下层土壤含水量全部蒸发，深层土壤蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为蒸发扩散系数，截距为下层土壤含水量；
<h5 style='pointer-events: none;'>Parameters</h5>

- water_content_upper: float | np.float64
        表示该时段初的上层土壤含水量，单位为mm。
- water_content_lower: float | np.float64 | np.float | np.float6464:
        表示该时段初的下层土壤含水量，单位为mm。
- precipitation: float | np.float64
        表示该时段初的降水量，单位为mm。
- evaporation_potential: float | np.float64
        表示该时段初的潜在蒸发量（流域蒸发能力），单位为mm。
- evaporation_coefficient: float | np.float64 **需要率定**
        表示蒸发扩散系数
- water_content_lower_max: float | np.float64 **需要率定**
        表示下层土壤蓄水容量
<h5 style='pointer-events: none;'>Return Values</h5>

- evaporation_upper: float | np.float64
        表示本时段的上层土壤蒸发量，单位为mm。
- evaporation_lower: float | np.float64
        表示本时段的下层土壤蒸发量，单位为mm。
- evaporation_deeper: float | np.float64
        表示本时段的深层土壤蒸发量，单位为mm。
- evaporation: float | np.float64
        表示本时段的总蒸发量，单位为mm。

```
def evaporate_3layer(water_content_upper: float | np.float64,
                     water_content_lower: float | np.float64,
                     precipitation: float | np.float64,
                     evaporation_potential: float | np.float64,
                     evaporation_coefficient: float | np.float64,  # 需要率定
                     water_content_lower_max: float | np.float64,  # 需要率定
                     ):
    # 如果上层土壤含水量和降水量之和大于等于潜在蒸发量
    if water_content_upper + precipitation >= evaporation_potential:
        # 上层土壤蒸发量为潜在蒸发量
        evaporation_upper = evaporation_potential
        # 下层土壤蒸发量为0
        evaporation_lower = 0
        # 深层土壤蒸发量为0
        evaporation_deeper = 0
    # 如果上层土壤含水量和降水量之和小于潜在蒸发量
    else:
        # 如果下层土壤含水比例大于蒸发扩散系数
        if water_content_lower >= evaporation_coefficient * water_content_lower_max:
            # 上层土壤蒸发量为上层土壤含水量和降水量之和
            evaporation_upper = water_content_upper + precipitation
            # 下层土壤蒸发量为下层土壤含水比例乘以扣除表层蒸发的潜在蒸发量
            evaporation_lower = (water_content_lower / water_content_lower_max) * (
                    evaporation_potential - evaporation_upper)
            # 深层土壤蒸发量为0
            evaporation_deeper = 0
        # 如果下层土壤含水比例小于蒸发扩散系数
        else:
            # 如果下层土壤含水比例大于蒸发扩散系数乘以潜在蒸发量减去上层土壤含水量和降水量之和
            if water_content_lower >= evaporation_coefficient * (
                    evaporation_potential - water_content_upper - precipitation):
                # 上层土壤蒸发量为上层土壤含水量和降水量之和
                evaporation_upper = water_content_upper + precipitation
                # 下层土壤蒸发量为蒸发扩散系数乘以扣除上层蒸发的潜在蒸发量
                evaporation_lower = evaporation_coefficient * (
                        evaporation_potential - evaporation_upper)
                # 深层土壤蒸发量为0
                evaporation_deeper = 0
            # 如果下层土壤含水比例小于蒸发扩散系数乘以潜在蒸发量减去上层土壤含水量和降水量之和
            else:
                # 上层土壤蒸发量为上层土壤含水量和降水量之和
                evaporation_upper = water_content_upper + precipitation
                # 下层土壤蒸发量为下层土壤含水量
                evaporation_lower = water_content_lower
                # 深层土壤蒸发量为蒸发扩散系数乘以扣除上层蒸发的潜在蒸发量，再减去下层土壤含水量
                evaporation_deeper = evaporation_coefficient * (
                        evaporation_potential - evaporation_upper) - evaporation_lower
    evaporation = evaporation_upper + evaporation_lower + evaporation_deeper
    return evaporation_upper, evaporation_lower, evaporation_deeper, evaporation
```
<h4 style='pointer-events: none;'> 2 Runoff Yield Model</h4>

 采用张力水容量曲线描述流域内张力水的变化，当流域内任意一点达到蓄水容量时，产流。
<h5 style='pointer-events: none;'> Parameters</h5>

- water_content: float | np.float64
        表示该时段初的土壤含水量，单位为mm。
- precipitation: float | np.float64
        表示该时段的降水量，单位为mm。
- evaporation: float | np.float64
        表示该时段的蒸发量，单位为mm。
- water_content_max_curve_max: float | np.float64 **计算得到**
        表示张力水容量曲线的最大值
- impermeable_area_ratio: float | np.float64 **需要率定**
        表示流域不透水面积比例
- water_content_max: float | np.float64 **需要率定**
        表示流域张力水容量曲线的最大值
- b: float | np.float64 **需要率定**
        表示张力水容量曲线公式中的指数
<h5 style='pointer-events: none;'> Return Values</h5>

- runoff: float | np.float64
        表示本时段的产流量，单位为mm。
- net_rainfall: float | np.float64
        表示本时段的净雨量，单位为mm。
```
def runoff_generate_stored_full(water_content: float | np.float64,
                                precipitation: float | np.float64,
                                evaporation: float | np.float64,
                                water_content_max_curve_max: float | np.float64,  # 计算得到
                                impermeable_area_ratio: float | np.float64,  # 需要率定
                                water_content_max: float | np.float64,  # 需要率定
                                b: float | np.float64,  # 需要率定
                                ):
    # # 计算流域张力水容量曲线的最大值
    # water_content_max_curve_max = water_content_max * (1 + b) / (1 - impermeable_area_ratio)
    # 计算流域特定蓄水量对应曲线的纵坐标
    a = water_content_max_curve_max * (1 - ((1 - water_content / water_content_max) ** (1 / (1 + b))))
    # 如果净雨量小于等于0
    net_rainfall = precipitation - evaporation
    if precipitation <= 0:
        # 产流量为0
        runoff = 0
    # 如果净雨量大于0
    else:
        # 如果没有全流域蓄满，即净雨量与a之和小于张力水容量曲线的最大值
        if a + net_rainfall < water_content_max_curve_max:
            # 产流量为张力水容量曲线在y轴上从a积分到a+P-E的面积
            runoff = net_rainfall + water_content - water_content_max + water_content_max * (
                    1 - (net_rainfall + a) / water_content_max_curve_max) ** (1 + b)
        # 如果全流域蓄满，即净雨量与a之和大于张力水容量曲线的最大值
        else:
            # 产流量为净雨量与a之和
            runoff = net_rainfall + water_content - water_content_max
    runoff = (1- impermeable_area_ratio) * runoff + impermeable_area_ratio * net_rainfall
    if runoff < 0:
        runoff = 0
    return runoff, net_rainfall
```
<h4 style='pointer-events: none;'>3 Soil Water Balance Model</h4>
<h5 style='pointer-events: none;'>Parameters</h5>

- water_content_upper: float | np.float64
        表示该时段初的上层土壤含水量，单位为mm。
- water_content_lower: float | np.float64:
        表示该时段初的下层土壤含水量，单位为mm。
- water_content_deeper: float | np.float64:
        表示该时段初的深层土壤含水量，单位为mm。
- evaporation_upper: float | np.float64
        表示该时段的上层土壤蒸发量，单位为mm。
- evaporation_lower: float | np.float64
        表示该时段的下层土壤蒸发量，单位为mm。
- evaporation_deeper: float | np.float64
        表示该时段的深层土壤蒸发量，单位为mm。
- precipitation: float | np.float64
        表示该时段的降水量，单位为mm。
- runoff: float | np.float64
        表示该时段的产流量，单位为mm。
- water_content_upper_max: float | np.float64 **需要率定**
        表示上层土壤蓄水容量
- water_content_lower_max: float | np.float64 **需要率定**
        表示下层土壤蓄水容量
- water_content_deeper_max: float | np.float64 **计算得到**
        表示深层土壤蓄水容量
<h5 style='pointer-events: none;'>Return Values</h5>

- water_content_upper: float | np.float64
        表示本时段末的上层土壤含水量，单位为mm。
- water_content_lower: float | np.float64:
        表示本时段末的下层土壤含水量，单位为mm。
- water_content_deeper: float | np.float64:
        表示本时段末的深层土壤含水量，单位为mm。
- water_content: float | np.float64:
        表示本时段末的土壤含水量，单位为mm。
        
```
def water_content_update(water_content_upper: float | np.float64,
                         water_content_lower: float | np.float64,
                         water_content_deeper: float | np.float64,
                         evaporation_upper: float | np.float64,
                         evaporation_lower: float | np.float64,
                         evaporation_deeper: float | np.float64,
                         precipitation: float | np.float64,
                         runoff: float | np.float64,
                         water_content_upper_max: float | np.float64,  # 需要率定
                         water_content_lower_max: float | np.float64,  # 需要率定
                         water_content_deeper_max: float | np.float64,  # 计算得到
                         ) -> object:
    # 上层土壤含水量更新
    water_content_upper = water_content_upper + precipitation - evaporation_upper - runoff
    # 下层土壤含水量更新
    water_content_lower = water_content_lower - evaporation_lower
    # 深层土壤含水量更新
    water_content_deeper = water_content_deeper - evaporation_deeper
    # 如果上层土壤蓄满
    if water_content_upper > water_content_upper_max:
        # 上层多余的水补充下层土壤
        water_content_lower += water_content_upper - water_content_upper_max
        # 上层土壤含水量即为上层土壤蓄水容量
        water_content_upper = water_content_upper_max
    # 如果下层土壤含水量大于下层土壤蓄水容量
    if water_content_lower > water_content_lower_max:
        # 下层多余的水补充深层土壤
        water_content_deeper += water_content_lower - water_content_lower_max
        # 下层土壤含水量即为下层土壤蓄水容量
        water_content_lower = water_content_lower_max
    # 如果深层土壤含水量大于深层土壤蓄水容量
    if water_content_deeper > water_content_deeper_max:
        # 深层土壤含水量即为深层土壤蓄水容量
        water_content_deeper = water_content_deeper_max
    water_content = water_content_upper + water_content_lower + water_content_deeper
    return water_content_upper, water_content_lower, water_content_deeper, water_content
```
<h4 style='pointer-events: none;'>4 Free Water Tank Model</h4>
<h5 style='pointer-events: none;'>Parameters</h5>

- runoff: float | np.float64
        表示该时段的产流量，单位为mm。
- net_rainfall: float | np.float64
        表示该时段的净雨量，单位为mm。
- free_water_content_last: float | np.float64
        表示上个时段末的自由水蓄水量，单位为mm。
- runoff_generate_area_ratio_last: float | np.float64
        表示上个时段末的产流面积比例。
- ex: float | np.float64 **需要率定**
        表示自由水蓄水容量曲线的最大值
- free_water_content_max: float | np.float64 **需要率定**
        表示流域自由水蓄水容量
- outflow_coefficient_inter: float | np.float64 **需要率定**
        表示壤中流出流系数
- outflow_coefficient_ground: float | np.float64 **需要率定**
        表示地下径流出流系数
<h5 style='pointer-events: none;'>Return Values</h5>

- runoff_surface: float | np.float64
        表示本时段的地表径流，单位为mm。
- runoff_inter: float | np.float64
        表示本时段的壤中流，单位为mm。
- runoff_ground: float | np.float64
        表示本时段的地下径流，单位为mm。
- free_water_content_next: float | np.float64
        表示本时段末的自由水蓄水量，单位为mm。
```
def water_division_3(runoff,
                     net_rainfall,
                     free_water_content_last,
                     runoff_generate_area_ratio_last,
                     ex,  # 需要率定
                     free_water_content_max,  # 需要率定
                     outflow_coefficient_inter,  # 需要率定
                     outflow_coefficient_ground,  # 需要率定
                     )
    # 如果净雨量为0，则所有径流成分都为0
    if runoff == 0:
        runoff_surface = 0
        runoff_inter = free_water_content_last * outflow_coefficient_inter * runoff_generate_area_ratio_last
        runoff_ground = free_water_content_last * outflow_coefficient_ground * runoff_generate_area_ratio_last
        free_water_content_next = free_water_content_last * (1 - outflow_coefficient_inter - outflow_coefficient_ground)
    # 如果净雨量不为0
    else:
        # 先求该时段的产流面积比例
        runoff_generate_area_ratio = runoff / net_rainfall  # 也就是自由水箱的底宽
        # 自由水蓄水容量曲线的最大值
        free_water_content_max_curve_max = free_water_content_max * (1 + ex)
        # 计算流域特定自由水蓄水量对应曲线的纵坐标
        # au = free_water_content_max_curve_max * (
        #         1 - (1 - free_water_content_last / free_water_content_max) ** (1 / (1 + ex)))  # 没有考虑FR的变化
        au = free_water_content_max_curve_max * (1 - (1 - (
                free_water_content_last / free_water_content_max * runoff_generate_area_ratio_last / runoff_generate_area_ratio)) ** (
                                                         1 / (1 + ex)))  # 考虑了FR的变化
        # 如果净雨量与au之和小于自由水蓄水容量曲线的最大值
        if net_rainfall + au < free_water_content_max_curve_max:
            runoff_surface = runoff_generate_area_ratio * (
                    net_rainfall + free_water_content_last * runoff_generate_area_ratio_last / runoff_generate_area_ratio - free_water_content_max + free_water_content_max * (
                    1 - (net_rainfall + au) / free_water_content_max_curve_max) ** (1 + ex))
        # 如果净雨量与au之和大于自由水蓄水容量曲线的最大值
        else:
            runoff_surface = runoff_generate_area_ratio * (
                    net_rainfall + free_water_content_last * runoff_generate_area_ratio_last / runoff_generate_area_ratio - free_water_content_max)
        # 计算本时段的自由水蓄量
        free_water_content = free_water_content_last * runoff_generate_area_ratio_last / runoff_generate_area_ratio + (
                runoff - runoff_surface) / runoff_generate_area_ratio
        # 对应的壤中流和地下径流
        runoff_inter = outflow_coefficient_inter * free_water_content * runoff_generate_area_ratio
        runoff_ground = outflow_coefficient_ground * free_water_content * runoff_generate_area_ratio
        # 计算下一时段的自由水蓄水容量
        free_water_content_next = free_water_content * (1 - outflow_coefficient_inter - outflow_coefficient_ground)
    return runoff_surface, runoff_inter, runoff_ground, free_water_content_next
```
<h4 style='pointer-events: none;'>5 Linear Reservoir Model</h4>
<h5 style='pointer-events: none;'>Parameters</h5>

- runoff_component: float | np.float64
        表示该时段的径流成分（地下径流、地面径流或壤中流），单位为mm。
- flow_last: float | np.float64
        表示上个时段末的流量，单位为m³/s。
- runoff_generate_area_ratio_last: float | np.float64
        表示上个时段末的产流面积比例。
- depletion_constant: float | np.float64 **需要率定**
        表示对应径流成分的消退系数，单位为1/d。
- unit_from_depth_2_volume: float | np.float64 **常量**
        表示从深度单位转化为体积单位的系数，单位为m/mm。
<h5 style='pointer-events: none;'>Return Values</h5>

- flow: float | np.float64
        表示演算的本时段流量，单位为m³/s。
```
def linear_reservoir(
        runoff_component: float | np.float64,
        flow_last: float | np.float64,
        runoff_generate_area_ratio_last: float | np.float64,  # 需要率定
        depletion_constant: float | np.float64,  # 需要率定
        unit_from_depth_2_volume: float | np.float64,  # 常量
):
    flow = (depletion_constant * flow_last +
            (1 - depletion_constant) * runoff_component * unit_from_depth_2_volume * runoff_generate_area_ratio_last)
    return flow
```