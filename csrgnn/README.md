<!--
 * @Descripttion: 
 * @Author: SijinHuang
 * @Date: 2022-03-17 16:02:29
 * @LastEditors: SijinHuang
 * @LastEditTime: 2022-05-29 10:38:59
-->
# SRGNN on Sepsis

A thesis project by Sijin Huang

## Environment

Python 3.8
The required packages are as follows:

```
torch==1.8.1
torch-geometric==1.7.0
scikit-learn
pyyaml
rich
pandarallel
tensorboard
```

## Data

Data preprocessing code was modified from Tucker's [code](https://github.com/ML4UWHealth/data-imputation). 
My modified version will be uploaded soon. Currently preprocessed data can be downloaded from Azure.
Download two preprocessed csv files from Azure `sepsisdata / Focused Features 2012-2019 / layers_2012_2019_preprocessed`
* layers_2012_2019_preprocessed_noimputation.csv
* layers_2012_2019_preprocessed.csv
Put csv files in `csrgnn/datasets`


After start training, extracted windows are stored with format:
1. `train.txt`
    * pickle of `[user_id_list, sequence_list, target_event_list, target_label_list]`
        * `id`: `user_id`
        * `sequence`: the sequence of records in the observation window (e.g. `[[21], [3, 37], [23, 43], [1, 5], [6], [23], [27, 7]]`)
        * `target_event`: event to predict given input sequence
        * `target_label`: binary ground truth (0 or 1) of `target_event`
2. `test.txt`
    * shares the same format as `train.txt`
    * 20% of the users sampled from the train set
3. `node_count.txt`
    * Stores the total number of nodes in the graph

## Arguments

|Name | Description| Default | Options |
|:---:|:---|:---:|:---|
| batch_size | The batch size to use during training and testing. | 100 | |
| hidden_size | The hidden state size. | 50 | [50,100] |
| epoch | The number of epochs to carrying out training for. | 30 | |
| lr | The learning rate of the optimizer. | 0.001 | [0.001, 0.0005, 0.0001] |
| l2 | Weight of the L2 regularization. | 1e-5 | [0.001, 0.0005, 0.0001, 0.00005, 0.00001] |
| lr_dc | The multiplicative factor of learning rate decay. | 0.5 | |
| lr_dc_step | The period of each learning rate decay. | 3 | |
| use_gat | DEPRECATED Whether to use GAT layers, if not InOutGGNN will be used. | False | [True, False] |

# start training

```bash
cd csrgnn/src/
python main.py --observe_window 12 --predict_window 24 --fold 1 --nrs ous --add_trend --add_layer3 --add_layer4
```