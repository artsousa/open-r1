CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 2.0e-6 \
    --num_train_epochs 5 \
    --max_seq_length 8192 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --logging_steps 5 \
    --save-steps 1000 \
    --output_dir /home/main/data/_Qwen2.5-Distill-w8
# 
#     --torch_dtype bf16 \
    # --max_steps 3000 \
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
