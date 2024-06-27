# Tiger finetuning
# <PATH TO BASE MODEL> the path to the base LLaMA-8B model
# <PATH TO DATASET> path to either the PJ, IPJ, or RPJ data you get from the Tiger-data-prep.ipynb, where you train
# Tiger-PJ-8B, Tiger-IPJ-8B, Tiger-Routing-8B respectively
# <PATH TO SAVE> the path to saving the checkpoints
# for validation eval, I find it goes CUDA OOM with A100 80G for long seq (~6K), so here I simply set eval_steps to
# large number to disable it, maybe you can turn it on with multicard setting, I'm too broke to try this myself :(
python sft.py \
    --base_model <PATH TO BASE MODEL> \
    --data_path <PATH TO DATASET> \
    --output_dir <PATH TO SAVE> \
    --val_size 300 \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --warmup_steps 10 \
    --eval_steps 10000 \
    --save_steps 40 \
    --save_total_limit 30 \
    --load_in_8bit \
    --learning_rate 3e-4 \
    --max_length 2048 \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj]' \
    --use_wandb \
    --wandb_project tiger_8b_sft \
    --wandb_run_name default_run