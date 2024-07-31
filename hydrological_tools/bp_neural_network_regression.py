import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader
from modules.cache import convert_dataframe_to_excel


@st.cache_data(show_spinner="读取数据..." if st.session_state.language == "中文" else "Reading data...")
def read_data(data):
    # 数据预处理
    data = pd.read_excel(data)
    # 去除空值
    data.dropna(inplace=True)
    return data


class BpDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


def choose_criterion_net(criterion):
    if criterion == "MSELoss":
        return nn.MSELoss()
    elif criterion == "L1Loss":
        return nn.L1Loss()
    elif criterion == "SmoothL1Loss":
        return nn.SmoothL1Loss()


def choose_scaler(scaler):
    if scaler == "MinMaxScaler":
        return MinMaxScaler()
    elif scaler == "StandardScaler":
        return StandardScaler()
    elif scaler == "MaxAbsScaler":
        return MaxAbsScaler()
    elif scaler == "RobustScaler":
        return RobustScaler()


def compute_loss(criterion, Y_pred, Y):
    if criterion == "MSELoss":
        return nn.functional.mse_loss(Y_pred, Y)
    elif criterion == "L1Loss":
        return nn.functional.l1_loss(Y_pred, Y)
    elif criterion == "SmoothL1Loss":
        return nn.functional.smooth_l1_loss(Y_pred, Y)


def bp_neural_network_regression():
    language = st.session_state["language"]
    st.subheader("bp神经网络回归" if language == "中文" else "bp neural network regression",
                 anchor=False)
    st.write("使用前馈神经网络进行回归" if language == "中文" else "Use feedforward neural network for regression")
    data = st.file_uploader(
        label="上传要回归的数据，数据形式为表格" if language == "中文" else "Upload data for regression, the data form is a table",
        type=['xlsx', 'xls'],
        key="data_for_bp_neural_network_regression",
        help="上传神经网络建模数据，数据格式为.xlsx或.xls" if language == "中文" else "Upload data for neural network modeling, "
                                                                                     "the data format is .xlsx or .xls",
    )
    with open("data/bp神经网络示例数据.xlsx", "rb") as file:
        st.download_button(
            label="下载bp神经网络示例数据" if language == "中文" else "Download bp neural network sample data",
            data=file,
            file_name='bp神经网络示例数据.xlsx' if language == "中文" else "bp neural network sample data.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            help="下载示例数据，按照示例数据的格式整理数据" if language == "中文" else "Download sample data and organize data according to the ",
        )
    if data is not None:
        data = read_data(data)
        columns = data.columns.tolist()
        col1, col2 = st.columns(2)
        # 设置数据归一化方法
        scaler = col1.selectbox(
            label="选择数据归一化方法" if language == "中文" else "Select data normalization method",
            options=["MinMaxScaler", "StandardScaler", "MaxAbsScaler", "RobustScaler"],
            index=0,
            key="scaler",
            help="MinMaxScaler：将数据缩放到[0, 1]之间\n\n"
                 "StandardScaler：将数据缩放到均值为0，方差为1的正态分布\n\n"
                 "MaxAbsScaler：将数据缩放到[-1, 1]之间\n\n"
                 "RobustScaler：将数据缩放到中位数为0，四分位数为1的正态分布" if language == "中文" else
            "MinMaxScaler: Scale data to [0, 1]\n\n"
            "StandardScaler: Scale data to a normal distribution with a mean of 0 and a variance of 1\n\n"
            "MaxAbsScaler: Scale data to [-1, 1]\n\n"
            "RobustScaler: Scale data to a normal distribution with a median of 0 and a quartile of 1",
        )
        # 创建一个selectbox，用于选择要回归的目标列
        target_column = col2.selectbox(
            label="选择要回归的目标列" if language == "中文" else "Select the target column to be interpolated",
            options=columns,
            index=len(columns) - 1,  # 默认选择最后一列
            key="target_column",
            help="该列的数据将作为回归的目标" if language == "中文" else "The data in this column will be used as the target of ",
        )
        # 创建一个multiselect，用于选择要回归的特征列
        feature_columns = st.multiselect(
            label="选择要回归的特征列" if language == "中文" else "Select the feature column to be interpolated",
            options=columns,
            default=columns[:-1],  # 默认排除最后一列
            key="feature_columns",
            help="这些列的数据将作为回归的特征" if language == "中文" else "The data in these columns will be used as the features of ",
        )
        # 设置数据集划分比例
        col1, col2 = st.columns(2)
        train_ratio = col1.slider(
            label="训练集比例" if language == "中文" else "Training set ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
            format="%.3f",
            key="train_ratio",
        )
        valid_ratio = col2.slider(
            label="验证集比例" if language == "中文" else "Validation set ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.3f",
            key="valid_ratio",
        )
        if train_ratio + valid_ratio >= 1:
            st.error(
                "训练集比例和验证集比例之和不能大于等于1" if language == "中文" else "The sum of the training set ratio and the "
                                                                                     "validation set ratio cannot be greater than "
                                                                                     "or equal to 1")
        else:
            # 数据归一化
            x = data[feature_columns].values
            y = data[target_column].values
            x_scaler = choose_scaler(scaler)
            y_scaler = choose_scaler(scaler)
            x = x_scaler.fit_transform(x)
            y = y_scaler.fit_transform(y.reshape(-1, 1)).squeeze()  # scaler要求二维数据
            # 划分数据集
            train_size = int(train_ratio * len(data))
            val_size = int(valid_ratio * len(data))
            test_size = len(data) - train_size - val_size
            st.write(f"测试集比例将设置为<font color=#00796B >**{1 - train_ratio - valid_ratio:.2f}**</font>，"
                     f"共有<font color=#00796B >**{data.shape[0]}**</font>个样本，"
                     f"其中<font color=#00796B >**{train_size}**</font>个样本作为训练集，"
                     f"<font color=#00796B >**{val_size}**</font>个样本作为验证集，"
                     f"<font color=#00796B >**{test_size}**</font>个样本作为测试集，"
                     "请定义网络的隐藏层参数，"
                     "第n行代表隐藏层中的第n层全连接层：" if language == "中文" else
                     f"The test set ratio will be set to <font color=#00796B >**{1 - train_ratio - valid_ratio:.2f}**</font>, "
                     f"there are <font color=#00796B >**{data.shape[0]}**</font> samples, "
                     f"of which <font color=#00796B >**{train_size}**</font> samples are used as training set, "
                     f"<font color=#00796B >**{val_size}**</font> samples are used as validation set, "
                     f"<font color=#00796B >**{test_size}**</font> samples are used as test set, "
                     "please define the parameters of the hidden layer of the network, "
                     "the nth row represents the nth fully connected layer in the hidden layer:",
                     unsafe_allow_html=True)
            x_train, x_val, x_test = x[:train_size], x[train_size:train_size + val_size], x[train_size + val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
            # 转换为张量(float32)
            x_train = torch.tensor(x_train, dtype=torch.float32)
            x_val = torch.tensor(x_val, dtype=torch.float32)
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            # 创建Dataset和DataLoader
            train_dataset = BpDataset(x_train, y_train)
            val_dataset = BpDataset(x_val, y_val)
            # test_dataset = BpDataset(x_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
            # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
            # 设置模型参数
            initial_data = {
                "神经元数量": [128, 64, 32],
                "激活函数": ["LeakyReLU", "LeakyReLU", "LeakyReLU"],
            } if language == "中文" else {
                "Number of neurons": [128, 64, 32],
                "Activation function": ["LeakyReLU", "LeakyReLU", "LeakyReLU"],
            }
            initial_data = pd.DataFrame(initial_data)
            edited_df = st.data_editor(
                initial_data,
                num_rows="dynamic",
                column_config={
                    "神经元数量": st.column_config.NumberColumn(
                        "神经元数量",
                        min_value=1,
                        max_value=256,
                        step=0,
                        required=True,
                        default=32,
                    ),
                    "激活函数": st.column_config.SelectboxColumn(
                        "激活函数",
                        options=["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"],
                        required=True,
                        default="LeakyReLU",
                    )
                } if language == "中文" else {
                    "Number of neurons": st.column_config.NumberColumn(
                        "Number of neurons",
                        min_value=1,
                        max_value=256,
                        step=0,
                        required=True,
                        default=32,
                    ),
                    "Activation function": st.column_config.SelectboxColumn(
                        "Activation function",
                        options=["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"],
                        required=True,
                        default="LeakyReLU",
                    )
                },
            )
            # 把edited_df神经元数量这一列的数据类型转换为int
            if language == "中文":
                edited_df["神经元数量"] = edited_df["神经元数量"].astype(int)
            else:
                edited_df["Number of neurons"] = edited_df["Number of neurons"].astype(int)

            # 创建模型
            def add_activation(layers, activation):
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                elif activation == "LeakyReLU":
                    layers.append(nn.LeakyReLU())
                elif activation == "Sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "Tanh":
                    layers.append(nn.Tanh())
                elif activation == "Softmax":
                    layers.append(nn.Softmax(dim=1))

            class BPNN(nn.Module):
                def __init__(self, df, input_size, output_size):
                    super(BPNN, self).__init__()
                    layers = [nn.Linear(input_size, df.iloc[0, 0])]
                    # 输入层
                    add_activation(layers, df.iloc[0, 1])
                    # 隐藏层
                    for i in range(1, len(df)):
                        layers.append(nn.Linear(df.iloc[i - 1, 0], df.iloc[i, 0]))
                        add_activation(layers, df.iloc[i, 1])
                    # 输出层
                    layers.append(nn.Linear(df.iloc[-1, 0], output_size))
                    # 将所有层组合起来
                    self.net = nn.Sequential(*layers)

                def forward(self, x):
                    return self.net(x)

            model = BPNN(edited_df, x_train.shape[1], 1)
            # 设置loss和优化器
            col1, col2, col3 = st.columns(3)
            criterion = col1.selectbox(
                label="损失函数" if language == "中文" else "Loss function",
                options=["MSELoss", "L1Loss", "SmoothL1Loss"],
                index=0,
                key="criterion",
                help="MSELoss：均方误差\n\n"
                     "L1Loss：平均绝对误差\n\n"
                     "SmoothL1Loss：平滑L1损失" if language == "中文" else
                "MSELoss: Mean squared error\n\n"
                "L1Loss: Mean absolute error\n\n"
                "SmoothL1Loss: Smooth L1 loss",
            )
            optimizer = col2.number_input(
                label="学习率" if language == "中文" else "Learning rate",
                min_value=0.0,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.6f",
                key="learning_rate",
                help="学习率的选择将影响模型的收敛速度" if language == "中文" else "The choice of learning rate will affect the convergence ",
            )
            # 训练模型
            epochs = col3.number_input(
                label="训练轮数" if language == "中文" else "Epochs",
                min_value=1,
                max_value=1000,
                value=200,
                step=1,
                key="epochs",
                help="训练轮数的选择将影响模型的精度" if language == "中文" else "The choice of epochs will affect the accuracy of the model",
            )
            # 用line_chart显示训练过程中的loss和val_loss
            # 用进度条显示训练进度
            train = st.toggle(
                label="训练模型" if language == "中文" else "Train model",
                value=False,
                key="train",
                help="训练模型将花费较长时间" if language == "中文" else "Training the model will take a long time",
            )
            if train:
                @st.cache_data(show_spinner="训练模型中..." if language == "中文" else "Training model...")
                def train_model(criterion, epochs, edited_df, optimizer, _train_loader, _val_loader, scaler):
                    # 根据用户选择的参数创建模型
                    model = BPNN(edited_df, x_train.shape[1], 1)
                    # 选择损失函数和优化器
                    criterion = choose_criterion_net(criterion)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer)
                    # 用line_chart显示训练过程中的loss和val_loss
                    st.divider()
                    col1, col2 = st.columns(2)
                    if language == "中文":
                        col1.write("标准化后训练集loss")
                        col2.write("标准化后验证集loss")
                    else:
                        col1.write("Normalized training set loss")
                        col2.write("Normalized validation set loss")
                    st.session_state.train_loss_values = []
                    st.session_state.val_loss_values = []
                    col1_holder = col1.empty()
                    col2_holder = col2.empty()
                    train_loss_chart = col1_holder.line_chart(use_container_width=True)
                    val_loss_chart = col2_holder.line_chart(use_container_width=True)
                    progress_bar = st.progress(0)
                    best_val_loss = float("inf")
                    for epoch in range(epochs):
                        # 训练模型
                        model.train()
                        train_loss_epoch = []
                        for X, Y in _train_loader:
                            optimizer.zero_grad()
                            pred = model(X)
                            loss = criterion(pred, Y.unsqueeze(1))
                            loss.backward()
                            optimizer.step()
                            train_loss_epoch.append([loss.item()])
                        train_loss_epoch_mean = np.mean(train_loss_epoch)
                        st.session_state.train_loss_values.append(train_loss_epoch_mean)
                        train_loss_chart.add_rows([train_loss_epoch_mean])
                        # 计算验证集的loss
                        model.eval()
                        val_loss_epoch = []
                        with torch.no_grad():
                            for X, Y in _val_loader:
                                pred = model(X)
                                loss = criterion(pred, Y.unsqueeze(1))
                                val_loss_epoch.append(loss.item())
                        val_loss_epoch_mean = np.mean(val_loss_epoch)
                        st.session_state.val_loss_values.append(val_loss_epoch_mean)
                        val_loss_chart.add_rows([np.mean(val_loss_epoch)])
                        # 保存最好的模型
                        if np.mean(val_loss_epoch) < best_val_loss:
                            best_model_state = model.state_dict().copy()
                        progress_bar.progress((epoch + 1) / epochs, text=f"epoch: {epoch + 1}/{epochs}")
                        # 重绘图表
                        col1_holder.line_chart(st.session_state.train_loss_values, use_container_width=True)
                        col2_holder.line_chart(st.session_state.val_loss_values, use_container_width=True)
                    return best_model_state

                best_model_state = train_model(criterion, epochs, edited_df, optimizer, train_loader, val_loader,
                                               scaler)
                # 测试模型
                # 如果st.session_state["best_model"]存在，说明已经训练过模型
                model.load_state_dict(best_model_state)
                model.eval()
                with torch.no_grad():
                    # 反归一化
                    y_pred = model(x_test)
                    y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()
                    y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
                    # 计算测试集的loss
                test_loss = compute_loss(criterion, y_pred.clone().detach(), y_test.clone().detach())
                test_loss_unscaled = compute_loss(criterion, torch.tensor(y_pred_unscaled),
                                                  torch.tensor(y_test_unscaled))
                # 生成一个DataFrame，用于显示测试结果
                if language == "中文":
                    test_result_unscaled = pd.DataFrame({
                        "真实值": y_test_unscaled,
                        "预测值": y_pred_unscaled,
                    })
                    test_result = pd.DataFrame({
                        "真实值": y_test.squeeze(),
                        "预测值": y_pred.squeeze(),
                    })
                else:
                    test_result_unscaled = pd.DataFrame({
                        "True value": y_test_unscaled,
                        "Predicted value": y_pred_unscaled,
                    })
                    test_result = pd.DataFrame({
                        "True value": y_test.squeeze(),
                        "Predicted value": y_pred.squeeze(),
                    })
                col1, col2 = st.columns(2)
                with col1:
                    st.write("标准化测试集结果" if language == "中文" else "Normalized test set results")
                    st.line_chart(test_result, color=["#0068c9", "#00796b"], use_container_width=True)
                    st.write(
                        f"标准化测试集loss为<font color=#00796B >**{test_loss.item():.4f}**</font>," if language == "中文" else
                        f"Normalized test set loss is <font color=#00796B >**{test_loss.item():.4f}**</font>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.write("未标准化测试集结果" if language == "中文" else "Unnormalized test set results")
                    st.line_chart(test_result_unscaled, color=["#0068c9", "#00796b"], use_container_width=True)
                    st.write(
                        f"未标准化测试集loss为<font color=#00796B >**{test_loss_unscaled.item():.4f}**</font>," if language == "中文" else
                        f"Unnormalized test set loss is <font color=#00796B >**{test_loss_unscaled.item():.4f}**</font>",
                        unsafe_allow_html=True,
                    )
                st.download_button(
                    label="下载测试集结果" if language == "中文" else "Download test results",
                    data=convert_dataframe_to_excel(pd.concat([test_result_unscaled, test_result], axis=1)),
                    file_name='bp神经网络回归测试结果.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help="下载测试结果",
                )
    st.divider()
    # 设置一个有问题的按钮
    st.caption(
        "程序有问题？请点击<a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>反馈文档</a>"
        if st.session_state["language"] == "中文" else
        "Something wrong? Please click <a href='https://docs.qq.com/doc/DU21SRXFYbXVBeWtF' style='color: #00796B;'>feedback document</a>",
        unsafe_allow_html=True
    )
