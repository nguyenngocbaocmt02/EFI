# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST

CACHE="/home/baocmt/.cache/huggingface/hub"
for MODEL in llama_7B alpaca_7B vicuna_7B llama2_chat_13B llama3_8B; do
    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --cache_dir $CACHE --dataset_name tqa_mc2
    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --cache_dir $CACHE --dataset_name tqa_gen_end_q
done