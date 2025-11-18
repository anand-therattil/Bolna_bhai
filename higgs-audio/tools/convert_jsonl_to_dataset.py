#!/usr/bin/env python3
"""
Convert JSONL format to Higgs Audio training dataset
Fixed version with correct paths and metadata generation
"""

import json
import os
import shutil
import torchaudio
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        return round(duration, 2)
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return 0.0

def extract_speaker_info(jsonl_path: str) -> Dict[str, str]:
    """Extract speaker information from file path"""
    file_name = Path(jsonl_path).stem
    
    # 根据文件名推断说话人信息
    if "tun" in file_name.lower():
        return {
            "speaker_id": "tun_speaker",
            "speaker_name": "Tun",
            "language": "zh",
            "gender": "unknown"
        }
    elif "huo" in file_name.lower():
        return {
            "speaker_id": "huo_speaker", 
            "speaker_name": "Huo",
            "language": "zh",
            "gender": "unknown"
        }
    else:
        return {
            "speaker_id": "unknown_speaker",
            "speaker_name": "Unknown",
            "language": "zh",
            "gender": "unknown"
        }

def analyze_emotion_from_text(text: str) -> str:
    """Simple emotion analysis based on text content"""
    text = text.lower()
    
    if any(word in text for word in ["？", "?", "什么", "哪", "怎么"]):
        return "questioning"
    elif any(word in text for word in ["！", "!", "好的", "是的", "对"]):
        return "affirming"
    elif any(word in text for word in ["超时", "错误", "失败"]):
        return "alerting"
    else:
        return "neutral"

def determine_scene(audio_path: str, text: str) -> str:
    """Determine scene based on audio path and text content"""
    if "phone" in audio_path.lower() or "电话" in text:
        return "phone_call"
    elif "meeting" in audio_path.lower() or "会议" in text:
        return "meeting_room"
    elif "录音" in text or "按键" in text:
        return "recording_system"
    else:
        return "quiet_room"

def convert_jsonl_to_dataset(
    jsonl_files: List[str],
    output_dir: str,
    copy_audio: bool = True,
    max_samples_per_speaker: int = None
):
    """Convert JSONL files to Higgs Audio dataset format"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dataset in: {output_path}")
    
    all_samples = []
    sample_counter = 0
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file}...")
        
        if not os.path.exists(jsonl_file):
            print(f"Error: JSONL file not found: {jsonl_file}")
            continue
        
        # 加载JSONL数据
        try:
            data = load_jsonl(jsonl_file)
        except Exception as e:
            print(f"Error loading JSONL file {jsonl_file}: {e}")
            continue
        
        # 获取说话人信息
        speaker_info = {
            "speaker_id": "1", 
            "speaker_name": "Harsh",
            "language": "hi",
            "gender": "male"
        }
        # speaker_info = extract_speaker_info(jsonl_file)
        speaker_id = speaker_info["speaker_id"]
        
        print(f"Found {len(data)} samples for speaker {speaker_id}")
        print(len(data))
        # 限制每个说话人的样本数量
        # if max_samples_per_speaker and len(data) > max_samples_per_speaker:
        #     data = data[:max_samples_per_speaker]
        #     print(f"Limited to {len(data)} samples for speaker {speaker_id}")
        
        processed_count = 0
        
        for idx, item in enumerate(tqdm(data, desc=f"Processing {speaker_id}")):
            try:
                print(item)
                # 解析消息
                messages = item.get("messages", [])
                audios = item.get("audios", [])
                
                if not messages or not audios:
                    continue
                
                # 找到用户消息和助手回复
                user_content = None
                assistant_content = None
                
                for msg in messages:
                    if msg["role"] == "user":
                        user_content = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_content = msg["content"]
                
                if not assistant_content or not audios:
                    continue
                
                # 获取音频文件路径
                original_audio_path = audios[0]
                if not os.path.exists(original_audio_path):
                    print(f"Audio file not found: {original_audio_path}")
                    continue
                
                # 生成新的文件名 - 直接放在输出目录根目录
                sample_id = f"{speaker_id}_{sample_counter:06d}"
                audio_filename = f"{sample_id}.wav"
                transcript_filename = f"{sample_id}.txt"
                
                # 音频文件路径 - 直接在输出目录根目录
                new_audio_path = output_path / audio_filename
                transcript_path = output_path / transcript_filename
                
                # 复制音频文件
                try:
                    if copy_audio:
                        shutil.copy2(original_audio_path, new_audio_path)
                    else:
                        # 创建符号链接
                        if new_audio_path.exists():
                            new_audio_path.unlink()
                        new_audio_path.symlink_to(os.path.abspath(original_audio_path))
                except Exception as e:
                    print(f"Error copying audio file {original_audio_path}: {e}")
                    continue
                
                # 创建转录文件
                try:
                    with open(transcript_path, 'w', encoding='utf-8') as f:
                        f.write(assistant_content.strip())
                except Exception as e:
                    print(f"Error writing transcript file {transcript_path}: {e}")
                    continue
                
                # 获取音频时长
                duration = get_audio_duration(str(new_audio_path))
                
                if duration == 0:
                    print(f"Skipping sample with invalid audio: {new_audio_path}")
                    # 清理已创建的文件
                    if new_audio_path.exists():
                        new_audio_path.unlink()
                    if transcript_path.exists():
                        transcript_path.unlink()
                    continue
                
                # 分析情感和场景
                emotion = analyze_emotion_from_text(assistant_content)
                scene = determine_scene(original_audio_path, assistant_content)
                
                # 创建样本元数据 - 使用文件名而不是相对路径
                sample_meta = {
                    "id": sample_id,
                    "audio_file": audio_filename,  # 直接使用文件名
                    "transcript_file": transcript_filename,  # 直接使用文件名
                    "duration": duration,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_info["speaker_name"],
                    "scene": scene,
                    "emotion": emotion,
                    "language": speaker_info["language"],
                    "gender": speaker_info["gender"],
                    "quality_score": 1.0,
                    "original_audio_path": original_audio_path,
                    "user_instruction": user_content,
                    "task_type": "audio_generation"
                }
                
                all_samples.append(sample_meta)
                sample_counter += 1
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing sample {idx} from {jsonl_file}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} samples for {speaker_id}")
    
    if not all_samples:
        print("Error: No valid samples were processed!")
        return False
    
    # 创建metadata.json
    total_duration = sum(s["duration"] for s in all_samples)
    avg_duration = total_duration / len(all_samples)
    
    metadata = {
        "dataset_info": {
            "total_samples": len(all_samples),
            "speakers": list(set(s["speaker_id"] for s in all_samples)),
            "languages": list(set(s["language"] for s in all_samples)),
            "total_duration": round(total_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "created_from": jsonl_files
        },
        "samples": all_samples
    }
    
    metadata_path = output_path / "metadata.json"
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully created metadata.json at: {metadata_path}")
    except Exception as e:
        print(f"Error creating metadata.json: {e}")
        return False
    
    print(f"\n✅ Dataset conversion completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"🎵 Total duration: {total_duration/3600:.2f} hours")
    print(f"⏱️  Average duration: {avg_duration:.2f} seconds")
    print(f"🎤 Speakers: {', '.join(metadata['dataset_info']['speakers'])}")
    
    # 统计各说话人样本数
    speaker_counts = {}
    for sample in all_samples:
        speaker_id = sample["speaker_id"] 
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
    
    print(f"\n📈 Samples per speaker:")
    for speaker, count in speaker_counts.items():
        print(f"   - {speaker}: {count} samples")
    
    # 验证生成的文件
    print(f"\n🔍 Verification:")
    audio_count = len(list(output_path.glob("*.wav")))
    txt_count = len(list(output_path.glob("*.txt")))
    print(f"   - Audio files: {audio_count}")
    print(f"   - Text files: {txt_count}")
    print(f"   - Metadata file: {'✅' if metadata_path.exists() else '❌'}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Higgs Audio dataset")
    
    parser.add_argument(
        "--jsonl_files", 
        nargs="+",
        default=[
            # "/root/code/new_work_code/HI-TransPA/swfit_workdir/fresh-little-lemon-workspace/data/swift_format/tun_audio.jsonl",
            "/root/code/new_work_code/HI-TransPA/swfit_workdir/fresh-little-lemon-workspace/data/swift_format/huo_audio.jsonl"
        ],
        help="Path to JSONL files"
    )
    
    parser.add_argument(
        "--output_dir",
        default="/root/code/higgs-audio-main/higgs_training_data_huo",
        help="Output directory for the dataset"
    )
    
    parser.add_argument(
        "--copy_audio",
        action="store_true",
        default=True,
        help="Copy audio files instead of creating symlinks"
    )
    
    parser.add_argument(
        "--max_samples_per_speaker",
        type=int,
        default=None,
        help="Maximum number of samples per speaker (None for no limit)"
    )
    
    parser.add_argument(
        "--clean_output_dir",
        action="store_true",
        help="Clean output directory before conversion"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"❌ Error: JSONL file not found: {jsonl_file}")
            return
    
    # 清理输出目录（如果指定）
    if args.clean_output_dir and os.path.exists(args.output_dir):
        print(f"🧹 Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # 转换数据集
    success = convert_jsonl_to_dataset(
        jsonl_files=args.jsonl_files,
        output_dir=args.output_dir,
        copy_audio=args.copy_audio,
        max_samples_per_speaker=args.max_samples_per_speaker
    )
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"You can now run training with:")
        print(f"python train_higgs_audio.py --train_data_dir {args.output_dir}")
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == "__main__":
    main()