# 如何写好论文的结果和讨论
根据上面的流程图，阅读两篇论文进行梳理。
## 1.RR-Former: Rainfall-runoff modeling based on Transformer 
### 1.1写作框架（分节）
本文按照同种模型的两种运用方式（研究方法）分节，分为了三节：Power of RR-former for individual rainfall-runoff modeling (Section 1)，Power of RR-former for individual rainfall-runoff modeling （Section 2），Limitations of RR-Former （Section 3）。

### 1.2写作段落（概述图表-论点-论据-解释原因）
For individual rainfall-runoff modeling, we compare our proposed 
RR-Former with two benchmarks including LSTM-MSV-S2S (Yin et al., 
2021) and LSTM-S2S (Xiang et al., 2020), and show their results of 7- 
day-ahead runoff predictions in Table 5 and more intuitively in Fig. 6.**（解释图表）** 
Besides, the CDFs of NSEs for 7-day-ahead runoff predictions provided 
by those three models are given in Fig. 7. **（解释图表）** The results show that for both 
the overall predictions (shown by NSE and RMSE) and the peak flow 
predictions (shown by ATPE-2%), the performance of our RR-Former is 
much **better** than that of the benchmarks. **（论点，比大小）** Besides, the performance of
7th-day-ahead runoff predictions provided by RR-Former is much better 
than the performance of 1-day-ahead runoff predictions provided by 
LSTM-S2S.**（论据，定量描述图表）** The median of NSEs of 5th-day-ahead runoff predictions 
provided by RR-Former is larger (better) than the median of NSEs of 1- 
day-ahead runoff predictions provided by LSTM-MSV-S2S. **（论据，定量描述图表）** Besides the 
accuracy of predictions, for the unbiasedness of predictions, the RRFormer is better than the benchmark models. **（论据，比性能）**

All these results show the power of RR-Former for individual rainfallrunoff modeling and our RR-Former entirely based on attention mechanisms can find a better representation of rainfall-runoff processing 
compared with the benchmarks based on the LSTM.**（论点，比大小）** Here, we take the 
basin numbered 12374250 as an example to show the advantages of RRFormer (see Fig. 8).**（解释图表）** We can see that our RR-Former has much better 
performance than the two benchmarks (i.e., LSTM-MSV-S2S and LSTMS2S) intuitively. In Fig. 8, our RR-Former estimates those several peak 
flows well, while the two benchmarks underestimate them heavily.**（解释图表）** This 
example intuitively shows that our RR-Former can provide a better 
representation of rainfall-runoff processing (better overall predictions 
and better peak flow predictions) compared with those two LSTM-based 
S2S models. 

As mentioned, our RR-Former contains two important processes: 
pretraining process and fine-tuning process. That is, our RR-Former 
achieves excellent performance on individual rainfall-runoff modeling 
task by pretraining using the so-called global dataset first and then finetuning the model for each basin using the so-called basin-specific 
dataset. Here, we want to show that these two processes are necessary by 
the RR-Former without fine-tuning and the RR-Former without pretraining. The RR-Former without fine-tuning uses only the global dataset 
to train the model and does not use the basin-specific dataset to fine-tune 
the model for each basin. The RR-Former without pretraining does not 
use the global dataset to pretrain the RR-Former but train one model for 
each basin by using its dataset (basin-specific dataset) directly. The 
statistic results provided by RR-Former, RR-Former without fine-tuning, 
and RR-Former without pretraining are shown in Table 6 and more 
intuitively in Fig. 9. **（解释图表）** Besides, the CDFs of NSEs for 7-day-ahead runoff predictions provided by RR-Former, RR-Former without fine-tuning, 
and RR-Former without pretraining are given in Fig. 10. **（解释图表）** The results 
show that both the pretraining process and the fine-tuning process are 
necessary for our RR-Former. **（论点，比大小）** Without either process, the performance of 
prediction accuracy and that of prediction unbiasedness decrease. **（论据，定量描述图表）**

The results show the power of our RR-Former for individual rainfallrunoff modeling and regional rainfall-runoff modeling. However, it is 
also worthwhile for pointing out its limitations.**（转折）** As a deep learning based 
data-driven model, the RR-Former learns the connection of two arbitrary positions and feature similarities of different levels from the data 
purely. **（解释原因）** The physical explanation is lacking to some extent, although we 
have tried our best to relate different terminologies from the RR-Former 
to interpretations from a hydrologic perspective.**（缺点也是论点）** Chadalawada et al. 
(2020) introduced a novel Machine Learning Rainfall-Runoff Model 
Induction Toolkit for hydrological model building using genetic programming and carried out a induction of lumped models. This kind of 
hydrologically informed machine learning approaches not only can 
generate the runoff predictions but also have meaningful hydrological 
The results show the power of our RR-Former for individual rainfallrunoff modeling and regional rainfall-runoff modeling. However, it is 
also worthwhile for pointing out its limitations. As a deep learning based 
data-driven model, the RR-Former learns the connection of two arbitrary positions and feature similarities of different levels from the data 
purely. The physical explanation is lacking to some extent, although we 
have tried our best to relate different terminologies from the RR-Former 
to interpretations from a hydrologic perspective. **（论据，用别的文献指出模型缺点）** Chadalawada et al. 
(2020) introduced a novel Machine Learning Rainfall-Runoff Model 
Induction Toolkit for hydrological model building using genetic programming and carried out a induction of lumped models. This kind of 
hydrologically informed machine learning approaches not only can 
generate the runoff predictions but also have meaningful hydrological 

## 2.Evaluation of Transformer model and Self-Attention mechanism in the Yangtze River basin runoff prediction 
### 2.1写作框架（分节）
Results部分按照研究方法（模型的不同）分为了两节，分别是Performances of LSTM, GRU and TSF for runoff prediction（Section 1）和Performances of LSTM, GRU and TSF for runoff prediction（Section 2）。Discussion部分按照影响因素（input step和Attention）分为了三节，分别是The TSF and SA model performance in the runoff prediction （Section 1）和The TSF and SA model performance in the runoff prediction （Section 2）和Limitations of this study and prospects for future studies （Section 3）。
### 2.2写作段落（概述图表-论点-论据-解释原因）
Fig. 5 shows the respective MSE of the LSTM, GRU and TSF models with different input time steps and prediction lengths. By 
combining the input time steps and prediction lengths that we chose, each model runs 6 times at the same prediction input time steps 
and runs 36 times in total.**（解释图表）** Generally, LSTM and GRU models show quite equivalent MSE, ranging from 0.2 × 10^-6 to 6 × 10^-6, for experiments with the same input time steps and prediction length, **（论据，比性能）** indicating their comparable performance in runoff prediction.  **（论点，比大小）**
Remarkably, the MSE from the TSF model is significantly increased, obviously larger than those from LSTM and GRU models, **（论据，比性能）** indicating the instinctively poorest performance of the TSF model in this study. **（论点，比大小）** For each fixed length of the input time step, obviously, the MSE values of all three models obviously become larger and larger as the increase of the prediction lengths, especially when the 
prediction lengths exceed 7d. **（论点，比大小）** This result meets the expectation that the runoff prediction becomes more challenging with the rise of 
prediction lengths. **（解释原因）** From the viewpoint of the input time steps, we can find that the length of input time steps does not affect much the 
performances of LSTM and GRU, **（解释原因）** as shown in Fig. 5(a) and (b), especially when it is relatively short, e.g., 1 d and 3 d. The longer the 
length of the input time steps is, the relatively better runoff prediction can be achieved, e.g., when the length is 5 d, 7 d, 9 d and 11 d, 
respectively. **（解释图表）** It is noteworthy that the longer input time steps do not always lead to better prediction results. **（论点，比大小）** However, the TSF model 
shows that increasing the length of the input time steps does not necessarily lead to improved prediction, as shown in Fig. 5(c). **（论据，比性能）**

To evaluate the capacities of these models in predicting extreme flooding events, we select one event with a runoff greater than 6.5 × 105 m3/s, which occurred in late August 2002, as an example. The observed runoff records and their predictions from LSTM, GRU and TSF models, with fixed input time steps of 25 d, are depicted in Fig. 6, showing the 1 d, 5 d and 9 d predictions, respectively, the 3 d, 7 d and 9 d result is showed in the Fig. S2.**（解释图表）** Apparently, with the increase in prediction lengths, runoff predictions from all models show a significant decline.**（论据，呈现数据）** When we make a 1 d prediction, the predicted runoff from all models is well consistent with each other and is very close to the observational peaks. **（论据，呈现数据）** Their NSE values are all above 0.995, **（论据，呈现数据）** indicating excellent prediction skills. **（论点，比大小）** On the other hand, when we make 5 d or 9d predictions, the predicted runoff from all models appears to deviate from the observed flood peaks, indicating reduced prediction skills. Especially for the 9 d prediction, all the models underestimate the flood peaks in late 
August, and the deviation reaches about 1.0 × 104 m3
/s. **（解释图表）** Based on the NSE values, the GRU model shows the best performance in all experiments of 1 d, 5 d and 9 d predictions; and the predictions of the TSF model are the worst with increasing deviations from the 
observed runoff from Fig. 6a–(c). Clearly, GRU ranks first and TSF ranks last in predicting flood events. 

From our knowledge, the TSF is a relatively new algorithm and has been found to perform a better result than the LSTM and GRU in many other disciplines, **（论点，比大小）** including computer science and time series prediction (He et al., 2021; Gan et al., 2021; Wu et al., 2021). However, in our study at Datong Station, the applications of both the LSTM and GRU model networks outperform the TSF in runoff prediction of the Yangtze River basin. **（论据，呈现数据）** The main reason for the weak performance of the TSF is the limited training information. The From our knowledge, the TSF is a relatively new algorithm and has been found to perform a better result than the LSTM and GRU in many other disciplines, including computer science and time series prediction (He et al., 2021; Gan et al., 2021; Wu et al., 2021). **（转折）** However, in our study at Datong Station, the applications of both the LSTM and GRU model networks outperform the TSF in runoff prediction of the Yangtze River basin. **（论据，呈现数据）** The main reason for the weak performance of the TSF is the limited training information. **（解释原因）**