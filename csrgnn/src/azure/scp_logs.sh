###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-22 13:07:47
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-27 13:45:30
### 
# scp -r /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log_csv/* uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log_csv/
# scp -r /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log_configs/* uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log_configs/
# scp -r /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log/* uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log/

rsync -av --progress /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log_csv/ uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log_csv
rsync -av --progress /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log_configs/ uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log_configs
rsync -av --progress /home/azureuser/sijin/sepsis/Sepsis-GNN/csrgnn/log/ uwr:/home/sijin/sepsis/Sepsis-GNN/csrgnn/log
