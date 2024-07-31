<h2 style='pointer-events: none;'>Base Infrastructure: interactive chart, normalization & metrics</h2>

Base infrastructure is a series of documents, mainly to write the common functions in deep learning for rainfall-runoff simulation into a package to achieve code reuse.

Because I found that some functions need to be rewritten every time I write the code, such as the loss of neural network, drawing the rainfall-runoff process chart, data normalization, etc., which are not difficult but time-consuming. So I plan to spend some time to write these into a package called ***utils***.
<h3 style='pointer-events: none;'>normalization</h3>

utils/data/normalize.py：
```
# 创建处理数据归一化的工具函数和类
from pandas import DataFrame, Series, concat
from typing import Union, Optional, Tuple, List


def normalize(data: Union[DataFrame, Series],
              use_cols: Optional[Union[str, List[str]]] = None,
              scale: Optional[List[float, float]] = None
              ) -> Tuple[Union[DataFrame, Series], DataFrame]:
    """
    将数据映射到指定范围内，对时间序列数据进行归一化处理。

    :param data: 时间序列数据
    :param use_cols: 需要归一化的列
    :param scale: 归一化范围，默认为[0, 1]，即将数据映射到0-1之间
    :return: 归一化后的数据和归一化参数
    """
    # 如果指定了需要归一化的列，则只对指定的列进行归一化
    if use_cols is not None:
        data = data[use_cols]
    # 如果没有指定归一化范围，则使用0-1作为默认范围
    if scale is None:
        scale = [0, 1]
    scale_lower_bound: float = scale[0]
    scale_upper_bound: float = scale[1]
    # 获取数据中每列的最大值、最小值
    min_series: Union[DataFrame, float] = data.min(axis=0)
    max_series: Union[DataFrame, float] = data.max(axis=0)
    # 对数据进行归一化处理
    data: Union[DataFrame, Series] = (data - min_series) / (max_series - min_series)
    # 对归一化之后的数据进行放缩
    data: Union[DataFrame, Series] = data * (scale_upper_bound - scale_lower_bound) + scale_lower_bound
    # 将最大值和最小值拼接成一个Dataframe，分别储存最大值、最小值、归一化范围
    # scale_params的行数等于data的列数，即特征数
    scale_params: DataFrame = concat([min_series, max_series], axis=1)
    scale_params.columns = ['min', 'max']
    scale_params['scale_lower_bound'] = scale_lower_bound
    scale_params['scale_upper_bound'] = scale_upper_bound
    return data, scale_params


def denormalize(data: Union[DataFrame, Series],
                scale_params: DataFrame,
                use_cols: Optional[Union[str, List[str]]] = None,
                ) -> Union[DataFrame, Series]:
    """
    将归一化后的数据还原到原始数据范围内。

    :param data: 归一化后的数据
    :param scale_params: 归一化参数，包含四列：最小值（min）、最大值（max）、归一化下界（scale_lower_bound）、归一化上界（scale_upper_bound）
    :param use_cols: 需要还原的列
    :return: 反归一化后的数据
    """
    # 如果指定了需要归一化的列，则只对指定的列进行归一化
    if use_cols is not None:
        try:
            data = data[use_cols]
            scale_params = scale_params.loc[use_cols]
        except KeyError:
            raise ValueError('在数据或归一化参数中挑选指定use_cols时出错')
    # 如果数据是DataFrame，即有多个特征，则需要判断数据的columns和归一化参数的index是否一致
    if isinstance(data, DataFrame) and not data.columns.equals(scale_params.index):
        raise ValueError('数据的columns和归一化参数的index不一致')
    # 如果数据是Series，即只有一个特征，则需要判断数据的name和归一化参数的index是否一致
    elif isinstance(data, Series) and not data.name == scale_params.index:
        raise ValueError('数据的name和归一化参数的index不一致')
    # 获取归一化参数
    min_series: Union[DataFrame, Series] = scale_params['min']
    max_series: Union[DataFrame, Series] = scale_params['max']
    scale_lower_bound: Union[DataFrame, Series] = scale_params['scale_lower_bound']
    scale_upper_bound: Union[DataFrame, Series] = scale_params['scale_upper_bound']
    # 对数据进行还原处理
    data: Union[DataFrame, Series] = (data - scale_lower_bound) / (scale_upper_bound - scale_lower_bound)
    data: Union[DataFrame, Series] = data * (max_series - min_series) + min_series
    return data


class Normalizer:
    def __init__(self,
                 scale_params: Optional[DataFrame] = None
                 ):
        self.scale_params: Optional[DataFrame] = scale_params

    def fit_transform(self,
                      data: Union[DataFrame, Series],
                      scale: [List[float, float]] = None,
                      use_cols: Optional[Union[str, List[str]]] = None
                      ) -> Union[DataFrame, Series]:
        """
        对数据进行归一化处理，并将归一化参数保存在self.scale_params中。

        :param data: 待归一化的数据
        :param scale: 归一化范围，默认为[0, 1]，即将数据映射到0-1之间
        :param use_cols: 需要归一化的列
        :return: 归一化后的数据
        """
        if scale is None:
            scale = [0, 1]
        self.scale_params, data = normalize(data, use_cols, scale)
        return data

    def inverse_transform(self, data: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
        """
        对数据进行反归一化处理。

        :param data:
        :return:
        """
        return denormalize(data, self.scale_params)

    def to_csv(self, path: str):
        """
        保存归一化参数到csv文件中。

        :param path: 保存路径
        :return:
        """
        self.scale_params.to_csv(path)

    def from_csv(self, path: str):
        """
        从csv文件中加载归一化参数。

        :param path: 加载路径
        :return:
        """
        self.scale_params = DataFrame.from_csv(path)
        return self.scale_params
```
<h3 style='pointer-events: none;'>interactive chart</h3>

utils/plot.py:
```
# 用Altair绘制各种数据图表
import altair as alt
import pandas as pd
from pandas import DataFrame, Series
from typing import Union, Optional


def check_if_single_column(data: Union[DataFrame, Series]) -> bool:
    """) -> bool:
    检查数据是否是Series或只有一列的DataFrame。

    :param data: 待检查的数据
    :return:  是否只有一列
    """
    if isinstance(data, Series):
        return True
    elif isinstance(data, DataFrame):
        if len(data.columns) == 1:
            return True
        else:
            return False
    else:
        return False


def melt_data(data: Union[DataFrame, Series],
              x_col: Optional[str] = None,
              ) -> DataFrame:
    """
    将数据转换为长格式，以适应altair的绘图要求。

    :param data: 待转换的数据
    :param x_col: x轴列名，如果为None，则使用索引作为x轴，如果data是Series或者单列DataFrame，则忽略此参数
    :return:  转换后的数据，一个三列的dataframe。x_col列为x轴，其它列名将转到"category"列，值将转到"y"列
    """
    if check_if_single_column(data):  # 如果是Series或只有一列的DataFrame
        data: DataFrame = data.reset_index()
        data = data.rename(columns={data.columns[0]: 'x'})
    else:
        if x_col is None:
            data = data.reset_index()
            data = data.rename(columns={data.columns[0]: 'x'})
        elif x_col not in data.columns:
            raise ValueError(f'x_col参数"{x_col}"不在数据的列名中')
        else:
            data = data.rename(columns={x_col: 'x'})
    return data.melt(id_vars='x', var_name='category', value_name='y')


def plot_line(data: Union[DataFrame, Series],
              x_col: Optional[str] = None,
              selector: bool = True,
              ) -> alt.Chart:
    """
    绘制单个或多个特征的折线图，支持交互式图表。

    :param data: 用于绘图的数据
    :param x_col: x轴列名，如果为None，则使用索引作为x轴
    :param selector: 选择器，用于交互式图表
    :return:  图表对象
    """
    # 将数据转换为长格式
    data = melt_data(data, x_col)
    # 检查x列的数据类型是否是时间
    x_is_time = True if data['x'].dtype == 'datetime64[ns]' else False
    # 如果x轴对应的数据是时间格式，则设置x轴的类型为时间
    x = alt.X('x', axis=alt.Axis(title=None,
                                 format='%Y-%m-%d'
                                 ) if x_is_time else alt.Axis(title=None)
              )
    # 设置y轴无标题
    y = alt.Y('y', axis=alt.Axis(title=None))
    # 设置悬停信息
    tooltip = [alt.Tooltip('x', title='x', format='%Y-%m-%d'
                           ) if x_is_time else alt.Tooltip('x', title='x'),
               alt.Tooltip('y', title='y')]
    # 绘制最基础的折线图
    chart = alt.Chart(data).mark_line().encode(x=x, y=y,
                                               color=alt.Color('category').title(''),
                                               tooltip=tooltip
                                               )
    # 透明选择器
    if selector:
        # 创建一个最近的选择器
        nearest = alt.selection_point(nearest=True, on='mouseover',
                                      fields=['x'], empty=False)
        selectors = alt.Chart(data).mark_point().encode(
            x=x,
            y=y,
            opacity=alt.value(0)
        ).add_params(nearest)  # 设置光标的x值
        # 绘制选择器选中的点
        points = chart.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
        # 显示文本
        text = chart.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'y:Q', alt.value(' '))
        )
        # 绘制选择器的位置
        rules = alt.Chart(data).mark_rule(color='gray').encode(
            x=x
        ).transform_filter(nearest)
        # 组合图表
        chart = alt.layer(chart, selectors, points, rules, text)
    chart = chart.properties(
        width=1000,
        height=500
    ).interactive()
    return chart


def plot_bar(data: Union[DataFrame, Series]) -> alt.Chart:
    """
    绘制单个特征的柱状图，支持交互式图表。

    :param data: 用于绘图的数据，index为x轴，values为y轴，只能设置为Series或单列DataFrame
    :return:  图表对象
    """
    # 数据格式转换
    if not check_if_single_column(data):
        raise ValueError('数据必须为Series或单列DataFrame')
    data = melt_data(data)
    # 绘制柱状图
    x = alt.X('x',
              axis=alt.Axis(title=None,
                            format='%Y-%m-%d %H:%M',
                            ) if data['x'].dtype == 'datetime64[ns]' else alt.Axis(title=None),
              scale=alt.Scale(padding=0)
              )
    y = alt.Y('y', axis=alt.Axis(title=None))
    tooltip = [alt.Tooltip('x', title='x', format='%Y-%m-%d %H:%M'
                           ) if data['x'].dtype == 'datetime64[ns]' else alt.Tooltip('x', title='x'),
               alt.Tooltip('y', title='y')
               ]
    chart = alt.Chart(data).mark_bar(size=8, binSpacing=0).encode(
        x=x,
        y=y,
        tooltip=tooltip
    )
    chart = chart.properties(
        width=1000,
        height=500
    ).interactive()
    return chart


def plot_runoff_and_rainfall(runoff: Union[DataFrame, Series],
                             rainfall: Union[DataFrame, Series],
                             title: Optional[str] = None,
                             ) -> alt.Chart:
    """
    绘制径流和降雨的折线图。

    :param runoff: 径流数据
    :param rainfall: 降雨数据
    :param title: 图表标题，推荐写NSE等评价指标
    :return:  图表对象
    """
    # 检查降水和径流数据的index是否是时间
    if not runoff.index.dtype == 'datetime64[ns]':
        raise ValueError('径流数据的index必须是时间')
    if not rainfall.index.dtype == 'datetime64[ns]':
        raise ValueError('降雨数据的index必须是时间')
    runoff_chart = plot_line(runoff, selector=False)
    rainfall_chart = plot_bar(rainfall)
    # 将runoff_chart和rainfall_chart的y轴都设置为[0, 原来的最大值的2倍]
    if isinstance(runoff, Series):
        runoff_max = runoff.max()
    else:
        runoff_max = runoff.max().max()
    rainfall_max = rainfall.max()
    runoff_chart = runoff_chart.encode(
        y=alt.Y('y',
                axis=alt.Axis(title=None),
                scale=alt.Scale(domain=[0, runoff_max * 2]),
                )
    )
    rainfall_chart = rainfall_chart.encode(y=alt.Y('y',
                                                   axis=alt.Axis(title=None, orient='right', ),
                                                   scale=alt.Scale(reverse=True, domain=[0, rainfall_max * 2]),
                                                   ))
    # 组合图表
    chart = alt.layer(runoff_chart, rainfall_chart).resolve_scale(y='independent').interactive(bind_y=False)
    # 设置图表标题
    if title is not None:
        chart = chart.properties(title=title)
    return chart


def plot_single_histogram(data: Union[DataFrame, Series]) -> alt.Chart:
    """
    绘制单个特征的直方图。

    :param data: 用于绘图的数据，只能设置为Series或单列DataFrame
    :return: 图表对象
    """
    if not check_if_single_column(data):
        raise ValueError('数据必须为Series或单列DataFrame')
    if isinstance(data, Series):  # 如果是Series，则转换为DataFrame
        if data.name is None:
            data.name = '这个特征没写名字'
        data = data.to_frame()
    x_name = data.name if isinstance(data, Series) else data.columns[0]
    x = alt.X(x_name, bin=alt.Bin(maxbins=40), scale=alt.Scale(padding=0), axis=alt.Axis(title=None))
    y = alt.Y('count()', axis=alt.Axis(title=None))
    tooltip = [alt.Tooltip('count()', title='count')]
    title = alt.TitleParams(text=f"{x_name}", anchor='middle')
    chart = alt.Chart(data).mark_bar(binSpacing=0).encode(x=x, y=y, tooltip=tooltip).properties(title=title,
                                                                                                width=1000,
                                                                                                height=250
                                                                                                ).interactive()
    return chart


def plot_histogram(data: Union[DataFrame, Series],
                   ) -> alt.Chart:
    """
    绘制特征的直方图。

    :param data: 用于绘图的数据
    :return:  图表对象
    """
    if isinstance(data, Series):
        return plot_single_histogram(data)
    else:
        chart_list = []
        for column in data.columns:
            chart_list.append(plot_single_histogram(data[column]))
        return alt.vconcat(*chart_list)
```
<h3 style='pointer-events: none;'>metrics</h3>

utils/metrics/metrics.py：
```
# 定义多个水文模型常用的损失函数的类
from torch import Tensor
from torch.nn import Module
from functions import rmse, nse, kge


class RMSELoss(Module):
    """
    创建一个均方根误差损失函数。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return rmse(y_pred, y_true)


class NSELoss(Module):
    """
    创建一个Nash-Sutcliffe效率系数损失函数。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return nse(y_pred, y_true)


class KGELoss(Module):
    """
    创建一个Kling-Gupta效率系数损失函数。
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return kge(y_pred, y_true)
```
utils/metrics/functions.py:
```
# 定义多个水文模型常用的损失函数
from torch import Tensor, sqrt, sum
from torch.nn.functional import mse_loss
from numpy import ndarray, float64, float32
from pandas import Series
from typing import Union, Sequence


def rmse(y_pred: Union[Tensor, ndarray, Series, Sequence],
         y_true: Union[Tensor, ndarray, Series, Sequence],
         ) -> Union[Tensor, float64, float32]:
    """
    计算均方根误差，用于评估模型的预测结果。
    RMSE越接近0，表示模型的预测结果越准确。
    RMSE = sqrt(mean((y_pred - y_true)^2))

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: 均方根误差
    """
    if isinstance(y_pred, Tensor):
        return sqrt(mse_loss(y_pred, y_true))
    else:
        return sqrt(mse(y_pred, y_true))


def nse(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算Nash-Sutcliffe效率系数，用于评估模型的预测结果。
    NSE越接近1，表示模型的预测结果越准确。
    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: Nash-Sutcliffe效率系数
    """
    return 1 - sum((y_pred - y_true) ** 2) / sum((y_true - y_true.mean()) ** 2)


def mse(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算均方误差，用于评估模型的预测结果。
    MSE越接近0，表示模型的预测结果越准确。
    MSE = mean((y_pred - y_true)^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: 均方误差
    """
    if isinstance(y_pred, Tensor):
        return mse_loss(y_pred, y_true)
    else:
        return ((y_pred - y_true) ** 2).mean()


def kge(y_pred: Union[Tensor, ndarray, Series, Sequence],
        y_true: Union[Tensor, ndarray, Series, Sequence],
        ) -> Union[Tensor, float64, float32]:
    """
    计算Kling-Gupta效率系数，用于评估模型的预测结果。
    KGE越接近1，表示模型的预测结果越准确。
    KGE = 1-sqrt((r-1)^2+(alpha-1)^2+(beta-1)^2)

    :param y_pred: 预测的序列
    :param y_true: 真实的序列
    :return: Kling-Gupta效率系数
    """
    y_pred_mean = y_pred.mean()
    y_true_mean = y_true.mean()
    y_pred_std = y_pred.std()
    y_true_std = y_true.std()
    alpha = y_pred_std / y_true_std
    beta = y_pred_mean / y_true_mean
    r = sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) / (y_pred_std * y_true_std)
    return 1 - sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
```