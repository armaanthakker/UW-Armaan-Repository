###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-27 00:31:18
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-27 00:40:48
### 

echo $(pwd)
mkdir -p ../log_zip
cp -r ../log ../log_zip/
cp -r ../log_configs ../log_zip/
cp -r ../log_csv ../log_zip/
cd .. && zip -rq log_zip.zip log_zip
