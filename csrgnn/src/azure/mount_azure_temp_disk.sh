###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-22 04:01:45
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-22 04:41:20
### 
rm -rf /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/datasets_graph
sudo chown -R azureuser /mnt
mkdir -p /mnt/sijin/datasets_graph
ln -s /mnt/sijin/datasets_graph /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/datasets_graph
