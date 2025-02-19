accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir /home/jupyter/data/Qwen2.5-Distill

#
# 
#    --packing \
#accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
#    --model_name_or_path meta-llama/Llama-3.2-1B \
#    --dataset_name open-r1/OpenR1-Math-220k \
#    --learning_rate 1.0e-5 \
#    --num_train_epochs 1 \
#    --packing \
#    --max_seq_length 4096 \
#    --per_device_train_batch_size 16 \
#    --gradient_checkpointing \
#    --bf16 \
#    --output_dir /home/jupyter/llama-3.1-Distill
