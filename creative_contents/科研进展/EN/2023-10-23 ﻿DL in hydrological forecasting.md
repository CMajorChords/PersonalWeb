
<h2 style='pointer-events: none;'>DL in hydrological forecasting</h2>
<h3 style='pointer-events: none;'>1.The effect of a variety of deep learning simulations of rainfall runoff</h3>

Mainly based on LSTM and TCN and their variants, from a large number of literature, Bi-LSTM effect is relatively better, the main models are:

LSTM: The role of LSTM is to extract information from several time steps ago as features into the current time step, and as time approaches, the greater the weight of the features, in other words, it is considered that the precipitation of the previous day is more important than the precipitation of the previous two days to the current runoff impact, which is intuitive in rainfall runoff simulations.

CNN-LSTM : Convolution extracts a local feature on an image, what is used to extract a time series here?

Bi-LSTM: Bi-directional LSTM actually corresponds to the extraction of context relations in NLP problems, that is, the first word and the next word of a certain word have an impact on the current word, which canbe clearly explained in the N LP problem, but if this network architecture is used for rainfall runoff simulation, all literature practices still only output the last time step of DecoderThe so-called Bi-LSTM degenerates into a simple LSTM plus a fully connected layer, but many literature points out that the effect of this is better than LSTM, I suspect that either the more fully connected layer has played a role, the network has become deeper, or the article is looking for the process along the results, and the randomness of the deep network training results itself is quite strong
<h3 style='pointer-events: none;'>2.Deep learning model interpretability method for rainfall runoff simulation</h3>

(1) After the model is trained, the model is interpreted as a black box

Permutation Importance: It is actually a weight interpretation method of queuing theory, mentioned in the undergraduate mathematical modeling method, which is essentially an objective empowerment method

Integrated gradient: Single-variable sensitivity analysis is very similar, except that the sensitivity analysis of general hydrological models usually looks at changes in NSE, and here is a new measurement formula:

$$
γ(α)=(1-α)x+αx
$$

Calculating the partial derivative of this formula for the x of the independent variable, you can get the contribution of the independent variable, but I don't understand why this formula can be explained.

(2) According to the neuronal weight of the LSTM layer, it is most important to calculate which day's precipitation (evaporation, runoff), this method only takes into account the LSTM layer, the so-called interpretability is more like storytelling, not recommended.

(3) Explainability module

While training the model to interpret, it seems to be the most powerful one, but there are many methods in this aspect, and there are many interpretable components added to it, and I don't understand much.
<h3 style='pointer-events: none;'>3.Independent variables that affect the model's performance</h3>

Runoff in the first few days is the most important driver, and as the foresight period increases, the importance of runoff decreases and the importance of precipitation increases

There are literature to use raster rainfall data as input to build a model, which is a good idea, but the display effect is not good, probably most of the raster data is interpolated by the point rainfall station, introducing too much interpolation caused by error and noise, here I think it is possible to use remote sensing precipitation products to be better
<h3 style='pointer-events: none;'>4.Modeling</h3>

There are literature to analyze the runoff in time series and then model the decomposed data one by one, but in fact, this scheme has the same effect as adding a fully connected layer at the end of the network, and it does not make much sense. In general, the biggest problem in the modeling of domestic literature here is that there is no feature engineering, that is, there is no suitable method to select the driving factors of the model, which is also related to the lack of data, there are too many data forms of hydrological data, there are time series, there are parameters in the form of constants, there are raster data and even raster data time series, it is difficult to make suitable feature selection, but for a single data form of the model, such as the input is all time series form of precipitation, runoff, To evaporate this model, you should find as many time series features as possible, such as air temperature and wind speed, and then use the feature engineering method to screen features to find suitable features for modeling.
<h3 style='pointer-events: none;'>5.Optimize</h3>

Most of the improved model optimization methods are still using non-gradient descent methods such as PSO and GA, note that unlike complex hydrological models, neural networks are derivable, and the derivative of each parameter of each neuron can be accurately obtained along the backpropagation

