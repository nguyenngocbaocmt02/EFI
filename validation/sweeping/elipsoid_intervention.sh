MODEL=llama_7B

# JUDGE=ft:davinci-002:ethicalytics::9t5UpZFx  ## NQOpen
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l ## Truthful
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
SAVE=""
CACHE=""
##################################################
#################   Ours   #######################
##################################################  
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
for MODEL in llama_7B; do
    for delta in 30.0; do
        for alpha in 0.2; do
            for lora_rank in 32; do
                for shrinking in 1.0; do
                    echo "model: $MODEL alpha_bl: $alpha, lora_rank: $lora_rank, delta: $delta, sh $shrinking"
                    CUDA_VISIBLE_DEVICES=3 python elipsoid_edit_layer.py --recal --shrinking $shrinking --delta $delta --alpha_bl $alpha --lora_rank $lora_rank --instruction_prompt default --exp_mode test --clf_only 0 --exp test2 --model_name $MODEL --device 0 --num_fold 2 --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --save_folder $SAVE --cache_dir $CACHE --loss_type cross_entropy
                    echo
                    echo
                done
            done
        done
    done
done
