import numpy as np
import pandas as pd
import streamlit as st
from modules.cache import convert_dataframe_to_excel
import altair as alt


@st.cache_data()
def NSE(simulation: np.ndarray,
        observation: np.ndarray,
        ):
    measurement_average = np.average(observation)
    return 1 - np.sum((simulation - observation) ** 2) / np.sum((observation - measurement_average) ** 2)


# @st.cache_data(show_spinner="绘制降雨径流过程图..." if st.session_state.language == "中文" else "plotting rainfall-runoff process...")
def plot_data(data, language):
    # 绘制径流折线图
    def flow_line(data: pd.DataFrame, column_name: str, color="#00796b"):
        return alt.Chart(data).mark_line(
            # 设置线宽
            strokeWidth=1.5,
            # 设置点大小
            point=alt.OverlayMarkDef(filled=True, size=20),

        ).encode(
            x=alt.X('time',
                    axis=alt.Axis(title='时间' if language == "中文" else 'time',
                                  # 设置刻度不可见
                                  tickOpacity=0,
                                  # 设置坐标轴为白色
                                  domainColor='#37474F',
                                  # 设置坐标轴宽度
                                  domainWidth=1,
                                  # 设置网格线
                                  grid=False,
                                  # 设置坐标轴可见
                                  labelOpacity=1,
                                  # 设置坐标轴标签为白色
                                  labelColor='#37474F',
                                  # 设置坐标轴标签格式为年月日时分
                                  format='%Y-%m-%d %H:%M'
                                  ),
                    ),
            y=alt.Y(column_name,
                    axis=alt.Axis(title='径流（m³/s）' if language == "中文" else 'flow (m³/s)',
                                  # 设置刻度不可见
                                  tickOpacity=0,
                                  # 设置坐标轴
                                  domainColor='#37474F',
                                  # # 设置网格线
                                  grid=True,
                                  # # 网格线
                                  gridColor='#37474F',
                                  # 设置y轴线宽度
                                  domainWidth=1,
                                  # 设置网格线宽度
                                  gridWidth=0.2,
                                  ),
                    # 设置y轴范围为原来的1.2倍
                    scale=alt.Scale(domain=[0, 2 * data[column_name].max()],
                                    ),
                    ),
            # 设置颜色
            color=alt.value(color),
            # 设置悬停信息
            tooltip=[alt.Tooltip('time', format='%Y-%m-%d %H:%M'), column_name],
        )

    # 绘制降水柱状图
    def precipitation_bar(data: pd.DataFrame, column_name: str):
        return alt.Chart(data).mark_bar(
            # 设置柱状图间隙
            binSpacing=0,
        ).encode(
            x=alt.X('time' if language == "中文" else 'time',
                    axis=alt.Axis(title='时间' if language == "中文" else 'time',
                                  # 设置刻度不可见
                                  tickOpacity=0,
                                  # 设置坐标轴
                                  domainColor='#37474F',
                                  # 设置坐标轴宽度
                                  domainWidth=1,
                                  # 设置网格线
                                  grid=False,
                                  # 设置坐标轴可见
                                  labelOpacity=1,
                                  # 设置坐标轴标签
                                  labelColor='#37474F',
                                  # 设置坐标轴标签格式为年月日时分
                                  format='%Y-%m-%d %H:%M'
                                  ),
                    ),
            y=alt.Y(column_name,
                    axis=alt.Axis(title='降水（mm）' if language == "中文" else 'precipitation (mm)',
                                  # 设置刻度不可见
                                  tickOpacity=0,
                                  # 设置坐标轴为白色
                                  domainColor='#37474F',
                                  # 设置y轴线宽度
                                  domainWidth=1,
                                  # 设置坐标轴可见
                                  labelOpacity=1,
                                  # 设置坐标轴标签为白色
                                  labelColor='#37474F',
                                  ),
                    scale=alt.Scale(reverse=True,
                                    domain=[0, 2.5 * data[column_name].max()]
                                    ),
                    ),
            # 设置颜色
            color=alt.value("#0068c9"),
            # 设置悬停信息，注意时间格式为年月日时分
            tooltip=[alt.Tooltip('time', format='%Y-%m-%d %H:%M'), column_name],
        )

    line_chart = flow_line(data, 'simulated flow') + flow_line(data, 'observed flow', color="#FF7F50")
    bar_chart = precipitation_bar(data, 'precipitation')

    chart = alt.layer(line_chart, bar_chart).resolve_scale(y='independent').configure_view(
        # # 设置背景颜色
        fill='#FFFFFF',
        # 设置图框颜色
        stroke='#37474F',
        # 设置图框宽度
        strokeWidth=1,
    ).interactive(bind_y=False)
    # 在streamlit中显示图表
    st.altair_chart(chart, use_container_width=True, theme=None)


# 三层蒸发模型
def evaporate_3layer(water_content_upper: float | np.float64,
                     water_content_lower: float | np.float64,
                     precipitation: float | np.float64,
                     evaporation_potential: float | np.float64,
                     evaporation_coefficient: float | np.float64,  # 需要率定
                     water_content_lower_max: float | np.float64,  # 需要率定
                     ):
    """
    三层蒸发模型

    - 当超过第一临界土壤含水量时：
        蒸发主要发生在土壤上层，上层土壤蒸发量即为潜在蒸发量，该阶段供水充分；
    - 当在第二和第一临界土壤含水量之间，扣除表层蒸发的潜在蒸发量比下层土壤能蒸发的少：
        上层土壤含水量和降水量全部蒸发，下层蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为下层土壤含水比例；
    - 当在第二和第一临界土壤含水量之间，扣除表层蒸发的潜在蒸发量比下层土壤能蒸发的多：
        上层土壤含水量和降水量全部蒸发，下层蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为蒸发扩散系数；
    - 当不超过第二临界土壤含水量时：
        降水量、上层下层土壤含水量全部蒸发，深层土壤蒸发量与扣除表层蒸发的潜在蒸发量成线性关系，系数为蒸发扩散系数，截距为下层土壤含水量；

    参数
    ----------
    water_content_upper: float | np.float64
        表示该时段初的上层土壤含水量，单位为mm。
    water_content_lower: float | np.float64 | np.float | np.float6464:
        表示该时段初的下层土壤含水量，单位为mm。
    precipitation: float | np.float64
        表示该时段初的降水量，单位为mm。
    evaporation_potential: float | np.float64
        表示该时段初的潜在蒸发量（流域蒸发能力），单位为mm。
    evaporation_coefficient: float | np.float64 **需要率定**
        表示蒸发扩散系数
    water_content_lower_max: float | np.float64 **需要率定**
        表示下层土壤蓄水容量

    返回值
    ----------
    evaporation_upper: float | np.float64
        表示本时段的上层土壤蒸发量，单位为mm。
    evaporation_lower: float | np.float64
        表示本时段的下层土壤蒸发量，单位为mm。
    evaporation_deeper: float | np.float64
        表示本时段的深层土壤蒸发量，单位为mm。
    evaporation: float | np.float64
        表示本时段的总蒸发量，单位为mm。
    """
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


# 蓄满产流模型
def runoff_generate_stored_full(water_content: float | np.float64,
                                precipitation: float | np.float64,
                                evaporation: float | np.float64,
                                water_content_max_curve_max: float | np.float64,  # 计算得到
                                impermeable_area_ratio: float | np.float64,  # 需要率定
                                water_content_max: float | np.float64,  # 需要率定
                                b: float | np.float64,  # 需要率定
                                ):
    """
    蓄满产流模型

    采用张力水容量曲线描述流域内张力水的变化，当流域内任意一点达到蓄水容量时，产流。

    参数
    ----------
    water_content: float | np.float64
        表示该时段初的土壤含水量，单位为mm。
    precipitation: float | np.float64
        表示该时段的降水量，单位为mm。
    evaporation: float | np.float64
        表示该时段的蒸发量，单位为mm。
    water_content_max_curve_max: float | np.float64 **计算得到**
        表示张力水容量曲线的最大值
    impermeable_area_ratio: float | np.float64 **需要率定**
        表示流域不透水面积比例
    water_content_max: float | np.float64 **需要率定**
        表示流域张力水容量曲线的最大值
    b: float | np.float64 **需要率定**
        表示张力水容量曲线公式中的指数

    返回值
    ----------
    runoff: float | np.float64
        表示本时段的产流量，单位为mm。
    net_rainfall: float | np.float64
        表示本时段的净雨量，单位为mm。
    """
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
    runoff = (1 - impermeable_area_ratio) * runoff + impermeable_area_ratio * net_rainfall
    if runoff < 0:
        runoff = 0
    return runoff, net_rainfall


# 土壤含水量更新
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
    """
    土壤含水量更新

    参数
    ----------
    water_content_upper: float | np.float64
        表示该时段初的上层土壤含水量，单位为mm。
    water_content_lower: float | np.float64:
        表示该时段初的下层土壤含水量，单位为mm。
    water_content_deeper: float | np.float64:
        表示该时段初的深层土壤含水量，单位为mm。
    evaporation_upper: float | np.float64
        表示该时段的上层土壤蒸发量，单位为mm。
    evaporation_lower: float | np.float64
        表示该时段的下层土壤蒸发量，单位为mm。
    evaporation_deeper: float | np.float64
        表示该时段的深层土壤蒸发量，单位为mm。
    precipitation: float | np.float64
        表示该时段的降水量，单位为mm。
    runoff: float | np.float64
        表示该时段的产流量，单位为mm。
    water_content_upper_max: float | np.float64 **需要率定**
        表示上层土壤蓄水容量
    water_content_lower_max: float | np.float64 **需要率定**
        表示下层土壤蓄水容量
    water_content_deeper_max: float | np.float64 **计算得到**
        表示深层土壤蓄水容量

    返回值
    ----------
    water_content_upper: float | np.float64
        表示本时段末的上层土壤含水量，单位为mm。
    water_content_lower: float | np.float64:
        表示本时段末的下层土壤含水量，单位为mm。
    water_content_deeper: float | np.float64:
        表示本时段末的深层土壤含水量，单位为mm。
    water_content: float | np.float64:
        表示本时段末的土壤含水量，单位为mm。
    """
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


# 自由水箱模型
def water_division_3(runoff,
                     net_rainfall,
                     free_water_content_last,
                     runoff_generate_area_ratio_last,
                     ex,  # 需要率定
                     free_water_content_max,  # 需要率定
                     outflow_coefficient_inter,  # 需要率定
                     outflow_coefficient_ground,  # 需要率定
                     ):
    """
    三水源划分：水箱模型

    参数
    ----------
    runoff: float | np.float64
        表示该时段的产流量，单位为mm。
    net_rainfall: float | np.float64
        表示该时段的净雨量，单位为mm。
    free_water_content_last: float | np.float64
        表示上个时段末的自由水蓄水量，单位为mm。
    runoff_generate_area_ratio_last: float | np.float64
        表示上个时段末的产流面积比例。
    ex: float | np.float64 **需要率定**
        表示自由水蓄水容量曲线的最大值
    free_water_content_max: float | np.float64 **需要率定**
        表示流域自由水蓄水容量
    outflow_coefficient_inter: float | np.float64 **需要率定**
        表示壤中流出流系数
    outflow_coefficient_ground: float | np.float64 **需要率定**
        表示地下径流出流系数
    返回值
    ----------
    runoff_surface: float | np.float64
        表示本时段的地表径流，单位为mm。
    runoff_inter: float | np.float64
        表示本时段的壤中流，单位为mm。
    runoff_ground: float | np.float64
        表示本时段的地下径流，单位为mm。
    free_water_content_next: float | np.float64
        表示本时段末的自由水蓄水量，单位为mm。
    """
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


# 线性水库模型
def linear_reservoir(
        runoff_component: float | np.float64,
        flow_last: float | np.float64,
        runoff_generate_area_ratio_last: float | np.float64,  # 需要率定
        depletion_constant: float | np.float64,  # 需要率定
        unit_from_depth_2_volume: float | np.float64,  # 常量
):
    """
    线性水库模型

    参数
    ----------
    runoff_component: float | np.float64
        表示该时段的径流成分（地下径流、地面径流或壤中流），单位为mm。
    flow_last: float | np.float64
        表示上个时段末的流量，单位为m³/s。
    runoff_generate_area_ratio_last: float | np.float64
        表示上个时段末的产流面积比例。
    depletion_constant: float | np.float64 **需要率定**
        表示对应径流成分的消退系数，单位为1/d。
    unit_from_depth_2_volume: float | np.float64 **常量**
        表示从深度单位转化为体积单位的系数，单位为m/mm。
    """
    flow = (depletion_constant * flow_last +
            (1 - depletion_constant) * runoff_component * unit_from_depth_2_volume * runoff_generate_area_ratio_last)
    return flow


def xinanjiang_model():
    # 显示标题
    st.subheader(
        "新安江模型" if st.session_state["language"] == "中文" else "XinAnJiang Model",
        anchor=False,
    )
    st.write(
        "使用三水源新安江模型进行降雨径流模拟" if st.session_state[
                                                      "language"] == "中文" else "Use Xinanjiang model for rainfall-runoff simulation"
    )
    # 读取数据
    data = st.file_uploader(
        label="上传新安江模型需要的数据" if st.session_state[
                                                "language"] == "中文" else "Upload data for Xinanjiang model",
        type=['xlsx', 'xls'],
        key="data_for_xinanjiang",
        help="上传新安江模型数据，数据格式为.xlsx或.xls",
    )
    if data is None:
        data = "data/新安江模型示例数据.xlsx"
    with open("data/新安江模型示例数据.xlsx", "rb") as file:
        st.download_button(
            label="下载新安江模型示例数据" if st.session_state[
                                                  "language"] == "中文" else "Download example data for Xinanjiang model",
            data=file,
            file_name='新安江模型示例数据.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="下载示例数据，按照示例数据的格式整理数据",
        )
    # 新安江模型
    # index列名为time
    data = pd.read_excel(data)
    data_length = len(data)
    precipitation = data['precipitation'].values
    evaporation_potential = data['evaporation potential'].values
    runoff_observed = data['observed flow'].values
    col1, col2 = st.columns(2)
    # 产流计算
    with col1.expander(
            "**产流模型**" if st.session_state["language"] == "中文" else "**Runoff generation model**",
            expanded=True,
    ):
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
        # st.markdown("<h4 style='pointer-events: none;'>产流模型</h4>" if st.session_state["language"] == "中文" else "<h4 style='pointer-events: none;'>Runoff generation model</h4>",
        #             unsafe_allow_html=True,
        #             )
        # 产流模型参数
        st.write("参数" if st.session_state["language"] == "中文" else "Parameters",
                 unsafe_allow_html=True, )
        # 构建一个dataframe，储存参数的名称、英文缩写、默认值
        runoff_producing_model_parameters = st.data_editor(
            pd.DataFrame({
                "abbreviation": ["WUM", "WLM", "WDM", "B", "C", "IM", "Kc"],
                "value": [20., 60., 40., 0.3, 0.18, 0.01, 0.8],
                "chinese name": ["上层土壤蓄水容量", "下层土壤蓄水容量", "深层土壤蓄水容量",
                                 "张力水容量曲线公式中的指数",
                                 "蒸发扩散系数", "流域不透水面积比例", "蒸发折算系数"],
                "english name": ["Upper soil water storage capacity", "Lower soil water storage capacity",
                                 "Deeper soil water storage capacity",
                                 "Exponent in the tension water capacity curve formula",
                                 "Evaporation diffusion coefficient", "Proportion of impervious area in the basin",
                                 "Evaporation conversion coefficient"],
            }),
            key="runoff_producing_model_parameters",
            hide_index=True,
            disabled=["chinese name", "english name", "abbreviation"],  # 仅仅value列是可编辑的
            column_config={
                "value": st.column_config.NumberColumn(
                    format="%.3f",
                    step=0.000001,  # 数值保留6位小数
                    min_value=0,
                )
            }
        )
        water_content_upper_max = runoff_producing_model_parameters["value"][0]
        water_content_lower_max = runoff_producing_model_parameters["value"][1]
        water_content_deeper_max = runoff_producing_model_parameters["value"][2]
        water_content_max = water_content_upper_max + water_content_lower_max + water_content_deeper_max
        B = runoff_producing_model_parameters["value"][3]
        water_content_max_curve_max = water_content_max * (1 + B)
        C = runoff_producing_model_parameters["value"][4]
        IM = runoff_producing_model_parameters["value"][5]
        Kc = runoff_producing_model_parameters["value"][6]
        evaporation_potential *= Kc
        # 初始值
        st.write("初始值" if st.session_state["language"] == "中文" else "Initial value",
                 unsafe_allow_html=True, )
        runoff_production_model_initial_value = st.data_editor(
            pd.DataFrame({
                "abbreviation": ["WU0", "WL0", "WD0"],
                "value": [0., 30., 40.],
                "chinese name": ["上层土壤含水量初始值", "下层土壤含水量初始值", "深层土壤含水量初始值"],
                "english name": ["Initial value of upper soil moisture", "Initial value of lower soil moisture",
                                 "Initial value of deeper soil moisture"],
            }),
            hide_index=True,
            key="runoff_production_model_initial_value",
            disabled=["chinese name", "english name", "abbreviation"],
            column_config={
                "value": st.column_config.NumberColumn(
                    format="%.3f",
                    step=0.000001,  # 数值保留6位小数
                    min_value=0,
                )
            }
        )
        water_content_upper[0] = runoff_production_model_initial_value["value"][0]
        water_content_lower[0] = runoff_production_model_initial_value["value"][1]
        water_content_deeper[0] = runoff_production_model_initial_value["value"][2]
        water_content[0] = water_content_upper[0] + water_content_lower[0] + water_content_deeper[0]
    # 汇流计算
    with col2.expander(
            "**汇流模型**" if st.session_state["language"] == "中文" else "**Routing mode**l",
            expanded=True,
    ):
        # st.write("<h4 style='pointer-events: none;'>汇流模型</h4>" if st.session_state["language"] == "中文" else "<h4 style='pointer-events: none;'>Routing model</h4>",
        #          unsafe_allow_html=True,)
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
        # 参数
        st.write("参数" if st.session_state["language"] == "中文" else "Parameters",
                 unsafe_allow_html=True, )
        # 构建一个dataframe，储存参数的名称、英文缩写、默认值
        routing_model_parameters = st.data_editor(
            pd.DataFrame({
                "abbreviation": ["SM", "EX", "F", "L", "KI", "KG", "CS", "CI", "CG"],
                "value": [20., 1.5, 290.1, 24., 0.6, 0.13, 0.09, 0.7, 0.98],
                "chinese name": ["流域自由水蓄水容量", "自由水容量曲线公式中的的指数", "流域面积", "时段长",
                                 "壤中流出流系数", "地下径流出流系数", "地表径流消退系数", "壤中流消退系数",
                                 "地下径流消退系数"],
                "english name": ["Free water storage capacity of the basin",
                                 "Exponent in the free water capacity curve formula",
                                 "Basin area", "Time lag", "Outflow coefficient of interflow",
                                 "Outflow coefficient of groundwater runoff", "Depletion coefficient of surface runoff",
                                 "Depletion coefficient of interflow", "Depletion coefficient of groundwater runoff"],
            }),
            key="routing_model_parameters",
            hide_index=True,
            disabled=["chinese name", "english name", "abbreviation"],
            column_config={
                "value": st.column_config.NumberColumn(
                    format="%.3f",
                    step=0.000001,  # 数值保留6位小数
                    min_value=0,
                    required=True,
                )
            }
        )
        free_water_content_max = routing_model_parameters["value"][0]
        EX = routing_model_parameters["value"][1]
        F = routing_model_parameters["value"][2]
        time_lag = routing_model_parameters["value"][3]
        U = F / (3.6 * time_lag)
        outflow_coefficient_inter = routing_model_parameters["value"][4]
        outflow_coefficient_ground = routing_model_parameters["value"][5]
        depletion_constant_surface = routing_model_parameters["value"][6]
        depletion_constant_inter = routing_model_parameters["value"][7]
        depletion_constant_ground = routing_model_parameters["value"][8]
        # 初始值
        st.write("初始值" if st.session_state["language"] == "中文" else "Initial value",
                 unsafe_allow_html=True, )
        routing_model_initial_value = st.data_editor(
            pd.DataFrame({
                "abbreviation": ["QS0", "QI0", "QG0", "Q0", "FR0", "S0"],
                "value": [10., 10., 10., 10., 0.1, 10.],
                "chinese name": ["地表径流初始值", "壤中流初始值", "地下径流初始值", "出口断面流量初始值",
                                 "产流面积比例初始值", "自由水蓄水量初始值"],
                "english name": ["Initial value of surface runoff", "Initial value of interflow",
                                 "Initial value of groundwater runoff", "Initial value of outlet flow",
                                 "Initial value of runoff generation area ratio",
                                 "Initial value of free water storage"],
            }),
            hide_index=True,
            key="routing_model_initial_value",
            disabled=["chinese name", "english name", "abbreviation"],
            column_config={
                "value": st.column_config.NumberColumn(
                    format="%.3f",
                    step=0.000001,  # 数值保留6位小数
                    min_value=0,
                )
            }
        )
        flow_surface[0] = routing_model_initial_value["value"][0]
        flow_inter[0] = routing_model_initial_value["value"][1]
        flow_ground[0] = routing_model_initial_value["value"][2]
        flow_total[0] = flow_surface[0] + flow_inter[0] + flow_ground[0]
        flow[0] = routing_model_initial_value["value"][3]
        runoff_generate_area_ratio[0] = routing_model_initial_value["value"][4]
        free_water_content[0] = routing_model_initial_value["value"][5]
    # 模型计算
    # st.write("<h4 style='pointer-events: none;'>模型计算</h4>" if st.session_state["language"] == "中文" else "<h4 style='pointer-events: none;'>Model calculation</h4>",
    #          unsafe_allow_html=True,)
    col1, col2, col3, _ = st.columns(4)
    calculate = col1.toggle(
        label="模型计算" if st.session_state["language"] == "中文" else "Model calculation",
        key="calculate",
        value=True,
    )
    if calculate:
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
                water_content=water_content[i - 1],
                precipitation=precipitation[i],
                evaporation=evaporation[i],
                water_content_max_curve_max=water_content_max_curve_max,
                impermeable_area_ratio=IM,
                water_content_max=water_content_max,
                b=B,
            )
            # 如果净雨量为负，产流面积比例为0
            if net_rainfall[i] <= 0:
                runoff_generate_area_ratio[i] = 0
            # 如果净雨量为正，产流面积比例为产流/净雨量
            else:
                runoff_generate_area_ratio[i] = runoff[i] / net_rainfall[i]
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
        download_data_result = col2.toggle(
            label="写入excel文件" if st.session_state["language"] == "中文" else "Write to excel file",
            value=False,
            key="download_data_result",
        )
        if download_data_result:
            # 将几段数据合并
            data_result = pd.DataFrame({
                "precipitation(mm)": precipitation,
                "observed flow(m³/s)": runoff_observed,
                "simulated flow(m³/s)": flow,
                "evaporation(mm)": evaporation,
                "evaporation potential(mm)": evaporation_potential,
                "runoff(mm)": runoff,
                "runoff generate area ratio": runoff_generate_area_ratio,
                "water content(mm)": water_content,
                "water content upper(mm)": water_content_upper,
                "water content lower(mm)": water_content_lower,
                "water content deeper(mm)": water_content_deeper,
                "runoff surface(mm)": runoff_surface,
                "runoff inter(mm)": runoff_inter,
                "runoff ground(mm)": runoff_ground,
                "free water content(mm)": free_water_content,
                "flow surface(m³/s)": flow_surface,
                "flow inter(m³/s)": flow_inter,
                "flow ground(m³/s)": flow_ground,
                "flow total(m³/s)": flow_total,
            }) if st.session_state["language"] == "English" else pd.DataFrame({
                "降水(mm)": precipitation,
                "实测流量(m³/s)": runoff_observed,
                "模拟流量(m³/s)": flow,
                "蒸发(mm)": evaporation,
                "流域蒸发能力(mm)": evaporation_potential,
                "产流(mm)": runoff,
                "产流面积比例": runoff_generate_area_ratio,
                "含水量(mm)": water_content,
                "上层土壤含水量(mm)": water_content_upper,
                "下层土壤含水量(mm)": water_content_lower,
                "深层土壤含水量(mm)": water_content_deeper,
                "地表径流产流量(mm)": runoff_surface,
                "壤中流产流量(mm)": runoff_inter,
                "地下径流产流量(mm)": runoff_ground,
                "自由水蓄水量(mm)": free_water_content,
                "地表径流流量(m³/s)": flow_surface,
                "壤中流流量(m³/s)": flow_inter,
                "地下径流流量(m³/s)": flow_ground,
                "总流量(m³/s)": flow_total,
            })
            data_result.index = data['time']
            data_result.index.name = "time" if st.session_state["language"] == "English" else "时间"
            # 下载数据
            col3.download_button(
                label="下载计算结果" if st.session_state["language"] == "中文" else "Download calculation result",
                data=convert_dataframe_to_excel(data_result),
                file_name='新安江模型计算结果.xlsx' if st.session_state[
                                                           "language"] == "中文" else "Xinanjiang model calculation result.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key="xinanjiang_model_result",
                help="下载计算结果",
            )
        # 绘图
        data_plot = pd.DataFrame({
            "time": data['time'],
            "precipitation": precipitation,
            "simulated flow": flow,
            "observed flow": runoff_observed,
        })
        # 计算效率系数
        st.write(
            "纳什效率系数：" if st.session_state["language"] == "中文" else "Nash efficiency coefficient:",
            f"<font color=#00796B >**{NSE(flow, runoff_observed):.3f}**</font>",
            unsafe_allow_html=True
        )
        plot_data(data_plot, language=st.session_state["language"])
    st.divider()
    # 设置一个有问题的按钮
    st.caption(
        "程序有问题？请点击<a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>反馈文档</a>"
        if st.session_state["language"] == "中文" else
        "Something wrong? Please click <a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>feedback document</a>",
        unsafe_allow_html=True
    )
