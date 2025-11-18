# 基本训练
python train_higgs_audio_lora.py \
    --train_data_dir /path/to/train/data \
    --eval_data_dir /path/to/eval/data \
    --output_dir ./output \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --use_lora \
    --lora_rank 16 \
    --fp16