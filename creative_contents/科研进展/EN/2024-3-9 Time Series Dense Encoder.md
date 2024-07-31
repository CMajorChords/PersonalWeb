<h2 style='pointer-events: none;'>Time Series Dense Encoder</h2>

[*Are Transformers Effectice for Time Series Forecasting?*](https://arxiv.org/pdf/2205.13504.pdf) questions the application of the attention mechanism of the transformer in time series forecasting. The paper argues that the position encoding in Transformer may lead to the loss of time information, and a linear model can be more effective than Transformer in long-term time series prediction tasks, based on which the DLinear model is proposed. [*Time Series Dense Encoder*](https://arxiv.org/pdf/2304.08424.pdf) is an extension of this view by Google, which uses residual blocks instead of the attention mechanism in Transformer. The results show that the model not only improves the training speed but also achieves SoTA on multiple training sets.
<h3 style='pointer-events: none;'>1. Model Structure</h3>

When linear models such as DLinear and NLinear have a linear dependency relationship between the horizon and lookback, they may achieve better results. However, DLinear and NLinear may not achieve better results mathematically and explain why they can achieve better results for nonlinear dynamic systems like 1. the runoff relationship between the previous and subsequent time steps, 2. the precipitation-runoff relationship. Such nonlinear systems require a structure that can describe the nonlinear relationship. Time Series Dense Encoder uses residual blocks to solve this problem.

Time Series Dense Encoder has 2 parts: encoder and decoder. The encoder is prefixed with a feature projection layer, and the decoder includes a stacking of residual blocks and a temporal decoder. The dense encoder and dense decoder can be combined into a single block. They are separated here just to adjust the hidden size of the blocks. The output dimension of the last dense layer represents the number of predicted time steps and the number of predicted features.
<h4 style='pointer-events: none;'>1.1 Encoding</h4>

The encoding task of the model is to map the covariates and lookback of the target sequence to a dense representation. The encoding part consists of two steps:

1. Feature Projection: The feature projection layers map the input features to a lower demension which is a hyperparameter that can be selected through cross-validation. This layer map the lookback and horizon of the covariates to a lower dimension to be input into the encoder with the lookback and horizon of the target. Note that this operation reduces the feature dimension rather than the time steps. The length of the flattened vector will be the sum of the length of the lookback and horizon multiplied by the number of features of the covariates if not reduced. Demension reduction will reduce the number of features. The operation of the feature projection layer can be written as:
$$
\tilde{\boldsymbol{x}}_t^{(i)}=\text{ResidualBlock}\Big(\boldsymbol{x}_t^{(i)}\Big).
$$
2. Dense Encoder: The dense encoder is a residual block that takes the output of the feature projection layer, static covariates, and the lookback of the target sequence as input and outputs an embedding. This embedding will be input into the decoder. The hidden layer parameters of n residual blocks in the dense encoder are all set to the same value, and the number of blocks is set to n<sub>e</sub>. The operation of the dense encoder can be represented as:
$$
\boldsymbol{e}^{(i)}=\text{Encoder}\Big(\mathbf{y}_{1:L}^{(i)};\tilde{\boldsymbol{x}}_{1:L+H}^{(i)};\boldsymbol{a}^{(i)}\Big)
$$
<h4 style='pointer-events: none;'>1.2 Decoding</h4>

The decoding task is to map the encoding representation to the future predicted time steps. The decoding part consists of two steps: Dense Decoder and Temporal Decoder.

1. Dense Decoder: The first part of the decoding is essentially a stacking of multiple residual blocks with the same hidden size. It takes the output $e^{(i)}$ of the encoding part as input and maps it to a single-layer vector $g^{(i)}$ of length $H\times p$, where p is the `decoderOutputDim`, and then reshapes $g^{(i)}$ to a matrix $D^{(i)}$ of size $d\times H$. The t-th column $d^{(i)}_t$ in the matrix can be considered as a decoded vector, which is used to predict the t-th time step. The operation can be represented as:
$$
\boldsymbol{g}^{(i)}=\text{Decoder}\Big(\boldsymbol{e}^{(i)}\Big)
$$
$$
\boldsymbol{D}^{(i)}=\text{Reshape}\Big(\boldsymbol{g}^{(i)}\Big)
$$

1. Temporal Decoder: The Temporal Decoder is used for the final prediction. It is a residual block with an output of 1, and each output corresponds to the prediction value of the t-th time step. The input of the residual block is the t-th column $d^{(i)}_t$ of the decoded matrix and the projected dynamic covariates $\tilde{\boldsymbol{x}}_{L+t}^{(i)}$. This operation establishes a 'highway' between the future covariates and the predicted value of the t-th time step. If the predicted time series has a particularly strong response to the corresponding time point of the covariates, this mapping will be very useful because it avoids the loss of information about these corresponding time points during the encoding and decoding process. The operation can be represented as:
$$
\hat{\boldsymbol{y}}_{L+t}^{(i)}=\text{TemporalDecoder}\Big(\boldsymbol{d}_t^{(i)};\tilde{\boldsymbol{x}}_{L+t}^{(i)}\Big)
$$

Finally, the model adds a residual connection between the lookback $y_{1:L}$ and the horizon $y_{L+1:L+H}$ of the target variable, which maps the lookback to a vector of size horizon.
<h3 style='pointer-events: none;'>2. Training</h3>

The model is trained using mini-batch gradient descent, where each batch consists of a batchSize number of time series and the corresponding lookback and horizon time points. Each epoch contains all the lookback and horizon pairs that can be constructed from the training period, i.e., two batches can have overlapping time points. This is the standard practice for all previous long-term prediction work.

The model is evaluated on the test set at each lookback-horizon pair, which is constructed from the test set. This is usually called rolling validation/evaluation. A similar evaluation can be used to adjust the model's selected parameters on the validation set.
<h3 style='pointer-events: none;'>3. Hyperparameters</h3>

The model's hyperparameters include the temporal decoder width `temporalWidth`, the number of hidden layers in the encoder and decoder `hiddenSize`, the number of layers in the encoder `numEncoderLayers`, the number of layers in the decoder `numDecoderLayers`, layer normalization `layerNorm`, dropout probability `dropout`, and learning rate `learningRate`. In the paper, the authors set the batch size to 512 and the `temporalWidth` to 4.