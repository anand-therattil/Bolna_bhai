#!/usr/bin/env python3
"""
Validate the converted Higgs Audio dataset
"""

import json
import os
import torchaudio
from pathlib import Path
import argparse

def validate_dataset(dataset_dir: str):
    """Validate the converted dataset"""
    
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.json"
    
    if not metadata_path.exists():
        print("Error: metadata.json not found!")
        return False
    
    # 加载metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    samples = metadata["samples"]
    print(f"Validating {len(samples)} samples...")
    
    valid_samples = 0
    invalid_samples = 0
    total_duration = 0
    
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        audio_file = dataset_path / sample["audio_file"]
        transcript_file = dataset_path / sample["transcript_file"]
        
        # 检查文件是否存在
        if not audio_file.exists():
            print(f"Missing audio file: {audio_file}")
            invalid_samples += 1
            continue
            
        if not transcript_file.exists():
            print(f"Missing transcript file: {transcript_file}")
            invalid_samples += 1
            continue
        
        # 检查音频文件
        try:
            waveform, sample_rate = torchaudio.load(str(audio_file))
            duration = waveform.shape[1] / sample_rate
            total_duration += duration
            
            # 检查音频长度
            if duration > 60:  # 超过60秒可能有问题
                print(f"Warning: Long audio ({duration:.2f}s): {audio_file}")
            
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}")
            invalid_samples += 1
            continue
        
        # 检查转录文件
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
                if not transcript:
                    print(f"Empty transcript: {transcript_file}")
                    invalid_samples += 1
                    continue
        except Exception as e:
            print(f"Error reading transcript {transcript_file}: {e}")
            invalid_samples += 1
            continue
        
        valid_samples += 1
        
        if (i + 1) % 100 == 0:
            print(f"Validated {i + 1}/{len(samples)} samples...")
    
    print(f"\nValidation Results:")
    print(f"- Valid samples: {valid_samples}")
    print(f"- Invalid samples: {invalid_samples}")
    print(f"- Success rate: {valid_samples/(valid_samples+invalid_samples)*100:.2f}%")
    print(f"- Total duration: {total_duration/3600:.2f} hours")
    print(f"- Average duration: {total_duration/valid_samples:.2f} seconds")
    
    return invalid_samples == 0

def main():
    parser = argparse.ArgumentParser(description="Validate Higgs Audio dataset")
    parser.add_argument(
        "--dataset_dir",
        default="/root/code/higgs-audio-main/higgs_training_data",
        help="Dataset directory to validate"
    )
    
    args = parser.parse_args()
    
    if validate_dataset(args.dataset_dir):
        print("✅ Dataset validation passed!")
    else:
        print("❌ Dataset validation failed!")

if __name__ == "__main__":
    main()