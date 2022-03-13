###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-12 08:01:51
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-13 09:21:04
### 
echo "Remember to comment out certain parameters in config.yml!"
progress_file="grid_progree.log"
echo "start grid seatch" > $progress_file
for i in {1..5}; do
    # for pred_wind in 3 6 12 24 48; do
    for pred_wind in 48 24 12 6 3; do
        for add_trend in "--add_trends" ""; do
            python main.py --observe_window 12 --predict_window $pred_wind --num_layers 2 $add_trend --only_preprocess
            for num_layers in 2 4 6; do
                echo "--observe_window 12 --predict_window $pred_wind --num_layers $num_layers $add_trend" >> $progress_file
                python main.py --observe_window 12 --predict_window $pred_wind --num_layers $num_layers $add_trend --skip_preprocess &
                sleep 5
            done
            wait
        done
    done
done
