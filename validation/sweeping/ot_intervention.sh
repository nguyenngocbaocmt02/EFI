MODEL=llama_7B

# JUDGE=ft:davinci-002:ethicalytics::9t5UpZFx  ## NQOpen
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l ## Truthful
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
SAVE="/big_storage/baonn/clf"
CACHE="/home/users/nus/binhnt/scratch/.cache/huggingface/hub"
##################################################
#################   Ours   #######################
##################################################  
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
for MODEL in llama_7B; do
    for rho in 0.025; do
            echo "model: $MODEL rho: $rho"
            CUDA_VISIBLE_DEVICES=2 python eff_edit_layer.py --rho $rho --lora_rank 4 --instruction_prompt default --exp_mode test --clf_only 0 --exp test --model_name $MODEL --device 0 --num_fold 2 --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --clf_folder $SAVE --cache_dir $CACHE --loss_type cross_entropy
            echo
            echo
    done
done

# python eff_edit_layer.py --lora_rank 4 --instruction_prompt default --exp_mode test --clf_only 0 --exp test --model_name llama_7B --device 0 --num_fold 2 --judge_name ft:davinci-002:ethicalytics:truthful:A0WsrZ0l --info_name ft:davinci-002:ethicalytics:truthful:A0WsrZ0l --eval_dataset truthful_qa --train_dataset truthful_qa --clf_folder /big_storage/baonn/clf --cache_dir /home/users/nus/binhnt/scratch/.cache/huggingface/hub --loss_type cross_entropy