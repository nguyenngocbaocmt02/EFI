MODEL=llama_7B
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l # TruthfulQA
#JUDGE=ft:davinci-002:ethicalytics::9t5UpZFx  ## NQOpen
#JUDGE=ft:davinci-002:ethicalytics::9t6np2Pi  ## TRivia
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
CACHE=""
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
##################################################
#################   ITI    #######################
##################################################
#TEST
for MODEL in llama_7B; do
    for alpha in 15; do
        for K in 48; do
            echo "model: $MODEL alpha: $alpha K: $K"
            CUDA_VISIBLE_DEVICES=0 python iti_edit_layer.py --instruction_prompt default --model_name $MODEL --use_mode test --num_heads $K --alpha $alpha --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --cache_dir $CACHE
            echo
            echo
        done
    done
done

