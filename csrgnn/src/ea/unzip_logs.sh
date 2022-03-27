###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-27 00:31:18
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-27 00:45:04
### 

echo $(pwd)
cd ..
unzip log_zip.zip
mv log_zip/log/* log/
mv log_zip/log_configs/* log_configs/
mv log_zip/log_csv/* log_csv/

