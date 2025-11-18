# # 处理单个文件
# python batch_generate_audio.py --input_path input.txt --output_dir outputs

# # 处理整个目录
# python batch_generate_audio.py --input_path text_files/ --output_dir outputs --batch_size 8

# 使用自定义参数
python examples/batch_generation.py \
    --model_path /root/code/higgs-audio-main/higgs_audio_huo_train-v4 \
    --input_path test/ \
    --output_dir huo_outputs \
    --batch_size 4 \
    --temperature 0.8 \
    --ref_audio belinda,chadwick