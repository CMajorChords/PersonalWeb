import pandas as pd
from pandas import Series, DataFrame
import streamlit as st


@st.cache_data(show_spinner="插值中..." if st.session_state.language == "中文" else "interpolating...")
def interpolate(dataset: Series | DataFrame, interval: str, method: str, start_time: str,):
    """
    按指定时间间隔插值数据集

    参数
    ----------
    dataset : Series | DataFrame
        待插值的数据集
    interval : str
        插值间隔，例如'1D'表示按天插值，'1H'表示按小时插值
    start_time : str
        插值起始时间，例如'2021-01-01'表示从2021年1月1日开始插值

    返回
    ----------
    dataset : Series | DataFrame
        插值后的数据集
    """
    # 生成插值后应该有的时间
    if isinstance(start_time, str):
        index_datetime = pd.date_range(start_time, dataset.index[-1], freq=interval)
    else:
        index_datetime = pd.date_range(dataset.index[0], dataset.index[-1], freq=interval)
    # 生成插值后的数据集
    dataset = dataset.reindex(index_datetime.union(dataset.index))
    # 插值
    dataset = dataset.interpolate(method=method)
    # 重新生成插值后应该有的时间
    dataset = dataset.reindex(index_datetime)
    return dataset


def time_series_interpolate():
    language = st.session_state["language"]
    st.subheader("时间序列插值" if language == "中文" else "Time series interpolation",
                 anchor=False
                 )
    data = st.file_uploader(
        label="上传要插值的数据，时间列名为：时间" if language == "中文" else "upload data for interpolation, the time column is named: Time",
        type=['xlsx', 'xls'],
        key="data_for_interpolation",
        help="上传要插值的数据，数据格式为.xlsx或.xls" if language == "中文" else "upload data for interpolation, data format is .xlsx or .xls",
    )
    with open("data/插值示例数据CN.xlsx" if language == "中文" else "data/插值示例数据EN.xlsx",
              "rb"
              ) as file:
        st.download_button(
            label="下载时间序列插值示例数据" if language == "中文" else "example data for time series interpolation",
            data=file,
            file_name='时间序列插值示例.xlsx' if language == "中文" else 'example for time series interpolation.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="下载示例数据，按照示例数据的格式整理数据" if language == "中文" else "example data for time series interpolation, organize the data according to the format of the example data",
        )
    # 如果用户上传了数据
    if data is not None:
        # 索引列名为“时间”
        try:
            @st.cache_data(show_spinner="读取数据..." if language == "中文" else "reading data...")
            def read_data(data, language):
                return pd.read_excel(data, index_col="时间" if language == "中文" else "Time")
            data = read_data(data, language=language)
            # 展示数据
            st.markdown("检查下数据吧：" if language == "中文" else "Check the data:")
            st.dataframe(data, height=300, use_container_width=True)
            # 选择插值间隔和插值起始时间
            col1, col2= st.columns(2)
            st.select_slider(
                label="选择插值间隔" if language == "中文" else "Select interpolation interval",
                options=["1min", "5min", "10min", "15min", "30min", "1H", "2H", "3H", "6H", "12H", "1D", "2D", "3D"],
                value='3H',
                key="interpolation_interval",
            )
            col1.text_input(
                    label="选择插值起始时间" if language == "中文" else "Select interpolation start time",
                    value=data.index[0],
                    key="interpolation_start_time",
            )
            col2.selectbox(
                label="选择插值方法" if language == "中文" else "Select interpolation method",
                options=["linear", "nearest", "zero", "quadratic", "cubic", "spline", "barycentric", "polynomial"],
                index=0,
                key="interpolation_method",
                help="linear: 线性插值,默认方法。\n\n"
                     "nearest: 最近邻插值，将待插值的值设置为最近的数据点的值。\n\n"
                     "zero: 零阶插值，将待插值的值设置为0。\n\n"
                     "quadratic: 二次多项式插值\n\n"
                     "cubic: 三次多项式插值\n\n"
                     "spline: 三次样条插值\n\n"
                     "barycentric: 重心插值" if language == "中文" else
                "linear: linear interpolation, the default method.\n\n"
                "nearest: nearest neighbor interpolation, set the value to be interpolated to the value of the nearest data point.\n\n"
                "zero: zero-order interpolation, set the value to be interpolated to 0.\n\n"
                "quadratic: quadratic polynomial interpolation\n\n"
                "cubic: cubic polynomial interpolation\n\n"
                "spline: cubic spline interpolation\n\n"
                "barycentric: barycentric interpolation"
            )
            # 选择是否插值
            st.toggle(
                label="插值" if language == "中文" else "interpolate",
                key="interpolate",
                value=False,
            )
            if st.session_state["interpolate"]:
                # 插值
                data = interpolate(data,
                                   interval=st.session_state["interpolation_interval"],
                                   method=st.session_state["interpolation_method"],
                                   start_time=st.session_state["interpolation_start_time"]
                                   )
                # 展示插值后的数据的
                st.markdown("插值后的数据的如下：" if language == "中文" else "The data after interpolation is as follows:")
                st.dataframe(data, height=300, use_container_width=True)
                # 下载插值后的数据
                st.download_button(
                    label="下载插值后的数据" if language == "中文" else "download data after interpolation",
                    data=data.to_csv().encode('utf-8-sig'),
                    file_name='插值后的数据.csv' if language == "中文" else 'data after interpolation.csv',
                    mime='text/csv' if language == "中文" else 'text/csv',
                    help="下载插值后的数据" if language == "中文" else "download data after interpolation",
                )
        # 有任何错误
        except Exception:
            st.error("数据格式错误，请按照示例数据的格式整理数据" if language == "中文" else "The data format is wrong, please organize the data according to the format of the example data")
    st.divider()
    # 设置一个有问题的按钮
    if language == "中文":
        st.caption("程序有问题？请点击<a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>反馈文档</a>",
                   unsafe_allow_html=True
                   )
    else:
        st.caption("Something wrong? Click <a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>feedback document</a>",
                   unsafe_allow_html=True
                   )
