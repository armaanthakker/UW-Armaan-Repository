<!--
 * @Descripttion: 
 * @Author: SijinHuang
 * @Date: 2022-03-17 16:02:29
 * @LastEditors: SijinHuang
 * @LastEditTime: 2022-03-20 10:49:07
-->
# SRGNN on Cues

## Environment

The required packages are as follows:

* PyTorch
* PyTorch-Geometric
* sklearn
* pyyaml
* rich
* pandarallel
* tensorboard

## Data

Train Period: 1 - 7 Jun
Test Period: 4 - 10 Jun

Raw Data:

1. `train.txt`
    * pickle of `[user_id_list, sequence_list, cue_list, y_list]`
        * `id`: `user_id`
        * `sequence`: the sequence of cues the user responded positively to over 7 days (e.g. `[[21], [3, 37], [23, 43], [1, 5], [6], [23], [27, 7]]`)
        * `cue`: the cues recommended to the user on the 8th day
        * `y`: the user's response to the recommended cues
2. `test.txt`
    * shares the same format as `train.txt`
    * 20% of the users sampled from the train set
3. `node_count.txt`
    * Stores the total number of nodes (cues) in the graph

## Arguments

|Name | Description| Default | Options |
|:---:|:---|:---:|:---|
| batch_size | The batch size to use during training and testing. | 100 | |
| hidden_size | The hidden state size. | 50 | [50,100] |
| epoch | The number of epochs to carrying out training for. | 10 | |
| lr | The learning rate of the optimizer. | 0.001 | [0.001, 0.0005, 0.0001] |
| l2 | Weight of the L2 regularization. | 1e-5 | [0.001, 0.0005, 0.0001, 0.00005, 0.00001] |
| lr_dc | The multiplicative factor of learning rate decay. | 0.5 | |
| lr_dc_step | The period of each learning rate decay. | 3 | |
| use_san | Whether to use self attention layers. | False | [True, False] |
| use_gat | Whether to use GAT layers, if not InOutGGNN will be used. | False | [True, False] |