# PeerCollaboration

## data processing

### ML-20M(https://grouplens.org/datasets/movielens/20m/):
```
This is a well-known benchmark dataset widely used for both traditional and sequential recommendation tasks~\cite{kang2018self,sun2019bert4rec,sun2020generic}. It contains around 20 million user-item interactions with 27,000 movies and 138,000 users.  Following the common practice in \cite{yuan2019simple, yuan2020future, kang2018self}, we assume that an observed feedback is available if an explicit rate is assigned to this item. We perform basic pre-processing to filter out the interactions with less than 5 users and users with less than 5 items to alleviate the effect of cold users and items. Then, we use timestamps to determine the order of interactions. Following~\cite{kang2018self, sun2019bert4rec}, we adopt the leave one out evaluation scheme. For each user, we hold out the last item of the interaction sequence as the test data, treat the item just before the last as the validation set, and utilize the remaining items for training. For the sequential recommendation task, we construct user's interaction sequences by using his recent $t$ interactions by the chronological order. For sequences shorter than t, we simple pad them with zero at the beginning of the sequence following~\cite{yuan2019simple}, while for sequences longer than t, we split them into several sub-sequences with length $t$ in the training set. In this paper, we set $t$ to 100 on this dataset.
```

### Retailrocket(https://www.kaggle.com/retailrocket/ecommerce-dataset): 
```
It is a public dataset collected from a real-world ecommerce website,  consisting  user shopping behaviors in 4.5 months. It contains 235,061 items and 1.4 million users. Similarly, we set $t$ to 10 to investigate recommendation performance for short-range interaction sequences.
```

## Implementation details
```
We train all models using the Adam optimizer on GPU. For common hyper-parameters, we consider the hidden dimension size (denoted by $d$) from \{16, 32, 64, 128, 256\} and the learning rate (denoted by $\eta$) from \{0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.005\}, the $L_2$ regularization coefficients from \{0.01, 0.001, 0.0005, 0.0001, 0.00005 0.00001\}, and dropout rate (denoted by $p$) from \{0, 0.1, 0.2, \dots, 0.9\} by grid search in the performance of the validation set. Specifically, we set the $d$ 256 for SASRec (except on Retailrocket), DNN (except on Retailrocket) and BPR. On Retailrocket, $d$ of SASRec and DNN is set to 64 to prevent overfitting. 
We use $\eta$ 1e-3 for SASRec and BPR, and 1e-4 for DNN on all datasets. In addition, we set batch size (denoted by $b$) to 128 for SASRec and DNN, and 2048 for BPR because of its enormous triple samples. As for model-specific hyper-parameters, we use two self-attention blocks (denoted by $l$) with one head for SASRec according to the original paper. Regarding DNN, we use one hidden layer on all datasets since using more layers does not lead to any improved results. Our PCRec uses exactly the same hyper-parameters (except $\eta$) as these individual base models. For $\eta$, one peer in PCRec uses exactly the same one with its base model, while the other peer uses a sub-optimal $\eta$. The model-specific hyper-parameter of PCRec  $\alpha$ is studied in the ablation study part. Without special mention, we report our results with the optimal $\alpha$.
```

## Layer-wise cooperation
Execute example:  
```
CUDA_VISIBLE_DEVICES=0 python3 -u SASRec_LW.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 1 --a 30 --seed 10 --cos --difflr --cooperation_type 'cooperation_type' > 1.log & CUDA_VISIBLE_DEVICES=1 python3 -u SASRec_LW.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 2 --a 30 --seed 11 --cos --difflr --cooperation_type 'cooperation_type' > 2.log  
```

The import configuration:  
```
--cooperation_type: This is used to control which cooperation type is used. There are 5 parameters in SASRec_layercooperation.py:  
alllayer_entropy：All layers are used by layer-wise cooperation with entropy criterion.
alllayer_eL1norm：All layers are used by layer-wise cooperation with L1-norm criterion.
onlyembed：Only the embedding layer uses layer-wise cooperation with entropy criterion.
onlymiddle：All layers exception the embedding and final layer use layer-wise cooperation with entropy criterion.  
onlyfinal： Only the final layer uses layer-wise cooperation with entropy criterion.
```


## Parameter-wise cooperation  
Execute example:  
```
CUDA_VISIBLE_DEVICES=0 python3 -u SASRec_PW.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 1 --percent 50 --seed 10 --cos --difflr > 1.log & CUDA_VISIBLE_DEVICES=1 python3 -u SASRec_PW.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 2 --percent 50 --seed 11 --cos --difflr > 2.log  
```


you can download a large sequential dataset of movielen-20m that has been pre-processed: https://drive.google.com/drive/folders/1TYtwwQruNcdDPQymsEgNRraXtjMf9jdl?usp=sharing

# References
https://github.com/kang205/SASRec code
