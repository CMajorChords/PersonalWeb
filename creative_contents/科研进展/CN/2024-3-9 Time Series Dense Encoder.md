<h2 style='pointer-events: none;'>Time Series Dense Encoder</h2>

[*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/pdf/2205.13504.pdf) 这篇文献对transformer的注意力机制在时间序列预测的应用提出了质疑。文献认为transformer的位置编码会导致时间信息的丢失，一个长期时间序列预测中只需要一个线性模型就可以比transformer更有效，并提出了DLinear模型。[*Time Series Dense Encoder*](https://arxiv.org/pdf/2304.08424.pdf)是谷歌在该观点中的进一步延伸，该模型使用残差块代替transformer的注意力机制，不仅改进了模型的训练速度，还在多个训练集上取得了SoTA。
<h3 style='pointer-events: none;'>1.模型结构</h3>

DLinear和NLinear等线性模型对预测值和lookback有线性依赖关系时，也许能够取得更好的效果。但是对于非线性的动态系统，如前后时间步的径流关系、降水径流关系，DLinear和NLinear无法在数学上达到并解释为什么能够取得更好的效果，诸如此类的非线性系统需要一个能描述非线性关系的结构。Time Series Dense Encoder使用了残差块来代替注意力机制或者线性模型。

Time Series Dense Encoder被分为了encoder和decoder两部分。encoder前缀了一个特征投影层，decoder包括了一系列的dense层，并后缀了一个temporal decoder。这里的dense encoder和dense decoder可以合并成一个单独的块。只是为了展示将他们分开以调整其中隐藏层的大小。最后一个dense layer的输出维度中，行数代表了预测的时间步，列数代表了预测的特征数。
<h4 style='pointer-events: none;'>1.1 Encoding</h4>

模型的encoding任务是将预测序列的过去部分（lookback）和协变量映射到一个dense representation。encoding部分氛围两个步骤：

1. 特征投影层：特征投影层将输入的特征映射到一个更高维度的空间。这个空间的维度是一个超参数，可以通过交叉验证来选择。这个层的目的是将协变量的lookback和horizon一起映射到一个更低的维度，从而和静态协变量和target的lookback一起输入到encoder中。注意，这一操作是降低特征维度而不是降低时间维度，如果不降维，展平后的长度将是lookback的长度加上horizon的长度之和乘以协变量的特征数，展平后将减少特征数量。特征投影层的运算可以写作：
$$
\tilde{\boldsymbol{x}}_t^{(i)}=\text{ResidualBlock}\Big(\boldsymbol{x}_t^{(i)}\Big).
$$
2. Dense Encoder：Dense Encoder是一个残差块，它将特征投影层的输出、静态协变量、预测序列的lookback作为输入，输出一个embedding。这个embedding将被输入到decoder中。Dense Encoder中的n个残差块隐藏层参数全部设置为相同的值，块的数量将被设置为n<sub>e</sub>。Dense Encider的作用可以用公式表示为：
$$
\boldsymbol{e}^{(i)}=\text{Encoder}\Big(\mathbf{y}_{1:L}^{(i)};\tilde{\boldsymbol{x}}_{1:L+H}^{(i)};\boldsymbol{a}^{(i)}\Big)
$$
<h4 style='pointer-events: none;'>1.2 Decoding</h4>

模型的decoding任务是将encoding representation映射到未来的几个预测时间步。decoding部分包括了两个步骤：Dense Decoder和Temporal Decoder。

1. Dense Decoder：第一个decoding部分，本质上还是具有相同隐藏层数量的多层残差块的堆叠。其接收encoding部分的输出$e^{(i)}$作为输入并将其映射到一个长度为$H\times p$的单层向量$g^{(i)}$中,这里的$p$指的是`decoderOutputDim`，然后将$g^{(i)}$reshape到一个$d\times H$的矩阵$D^{(i)}$。该矩阵中的第t列$d^{(i)}_t$可以被认为是一个已解码向量，该已解码向量被用作预测第t个时间步。整个的操作用公式表示为：
$$
\boldsymbol{g}^{(i)}=\text{Decoder}\Big(\boldsymbol{e}^{(i)}\Big)
$$  
$$
\boldsymbol{D}^{(i)}=\text{Reshape}\Big(\boldsymbol{g}^{(i)}\Big)
$$

2. Temporal Decoder：Temporal Decoder用于做最后的预测。temporal decoder是一个输出为1的残差块，每一个输出对应着第$t$个时间步的预测值。该残差块输入为Dense输出的解码矩阵的第t列$d^{(i)}_t$和经过投影的动态协变量$\tilde{\boldsymbol{x}}_{L+t}^{(i)}$。这一操作相当于为从第t个时间步的未来协变量到第t个时间步的预测值之间的关系建立了一个快速映射通道。如果预测的时间序列与对应时间点的协变量有特别强烈的响应，那么这个映射将非常有用，因为它能避免包含这些对应时间点的信息在encoding和decoding的过程中丢失。该操作用公式可以表示为：
$$
\hat{\boldsymbol{y}}_{L+t}^{(i)}=\text{TemporalDecoder}\Big(\boldsymbol{d}_t^{(i)};\tilde{\boldsymbol{x}}_{L+t}^{(i)}\Big)
$$
最后，模型在目标变量的lookback$y_{1:L}$和horizon
$y_{L+1:L+H}$之间加了一个残差连接，该残差连接映射lookback到一个size为horizon的向量。
<h3 style='pointer-events: none;'>2.模型训练</h3>

该模型使用小批量梯度下降进行训练，其中每个批次由一个 batchSize 数量的时间序列以及相应的回溯和水平时间点组成。每个 epoch 都包含所有可以从训练期构建的回溯和地平线对，即两个小批量可以有重叠的时间点。这是以前所有长期预测工作的标准做法。

该模型在每个lookback-horizon对上的测试集上进行评估，这些测试集从测试集构建。这通常称为滚动验证/评估。可以选择使用对验证集的类似评估来调整模型选择的参数。
<h3 style='pointer-events: none;'>3.超参数</h3>

模型的超参数包括temporal decoder宽度`temporalWidth`，encoder和decoder的隐藏层数量`hiddenSize`，encoder的层数`numEncoderLayers`，decoder的层数`numDecoderLayers`，层归一化`layerNorm`，dropout概率`dropout`，学习率`learningRate`。在论文中，作者设置了batchsize为512，'teporalWidth'为4。