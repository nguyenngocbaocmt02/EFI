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
    for kappa in 0.01 0.05 0.1; do
            for lora_rank in 32; do
            echo "model: $MODEL kappa: $kappa lora_rank: $lora_rank"
            CUDA_VISIBLE_DEVICES=3 python subspace_edit_layer.py --recal --kappa $kappa --lora_rank $lora_rank --instruction_prompt default --exp_mode test --clf_only 0 --exp max_min_version --model_name $MODEL --device 0 --num_fold 2 --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --save_folder $SAVE --cache_dir $CACHE --loss_type cross_entropy
            echo
            echo
        done
    done
done
