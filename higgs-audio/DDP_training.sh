torchrun --nproc_per_node=2 trainer/trainer_ddp.py \
  --model_path /root/code/higgs-audio-main/model_ckpt \
  --audio_tokenizer_path /root/code/higgs-audio-main/model_ckpt_tokenizer \
  --train_data_dir /root/code/higgs-audio-main/higgs_training_data_mini_vn_10k \
  --output_dir ./output/my_multi_gpu_run \
  --per_device_train_batch_size 4 \
  --num_train_epochs 2 \
  --use_lora \
  --bf16