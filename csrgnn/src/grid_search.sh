###
 # @Descripttion: 
 # @Author: SijinHuang
 # @Date: 2022-03-12 08:01:51
 # @LastEditors: SijinHuang
 # @LastEditTime: 2022-03-22 05:06:53
### 
echo "Remember to comment out certain parameters in config.yml!"
progress_file="grid_progress.log"
echo "start grid seatch. Date: $(TZ=Asia/Shanghai date)" >> $progress_file
for fold in {1..5}; do
    # for pred_wind in 3 6 12 24 48; do
    for pred_wind in 48 24 12 6 3; do
        for add_trend in "--add_trends" ""; do
            for no_imputation in "--no_imputation" ""; do
                DATA_ARGS="--observe_window 12 --predict_window $pred_wind --fold $fold --nrs nds $add_trend $no_imputation"
                python main.py $DATA_ARGS --only_preprocess
                for num_layers in 2 4 6; do
                    TRAIN_ARGS="$DATA_ARGS --num_layers $num_layers"
                    echo "Args: $TRAIN_ARGS Date: $(TZ=Asia/Shanghai date)" >> $progress_file
                    CUDA_VISIBLE_DEVICES=0 python main.py $TRAIN_ARGS --skip_preprocess &
                    sleep 5
                done
                wait
            done
        done
    done
done
