#!/usr/bin/env python3
"""
Script to create JSONL file from CSV data for audio transcription.
Each line in the JSONL file contains the audio path and transcription.
"""

import csv
import json
import os
from pathlib import Path


def create_jsonl_from_csv(csv_file, audio_dir, output_file):
    """
    Create a JSONL file from CSV data and audio files.
    
    Args:
        csv_file: Path to the CSV file with format: filename, transcription
        audio_dir: Directory containing the audio files
        output_file: Path to the output JSONL file
    """
    # Counter for processed entries
    processed = 0
    skipped = 0
    
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        with open(csv_file, 'r', encoding='utf-8') as csv_input:
            # Read CSV with tab delimiter
            reader = csv.reader(csv_input, delimiter='\t')
            
            for row in reader:
                if len(row) != 2:
                    print(f"Warning: Skipping malformed row: {row}")
                    skipped += 1
                    continue
                
                filename, transcription = row
                
                # Replace .txt extension with .wav
                audio_filename = filename.replace('.txt', '.wav')
                audio_path = os.path.join(audio_dir, audio_filename)
                
                # Check if audio file exists
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    skipped += 1
                    continue
                
                # Create the JSON object
                entry = {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": f"<audio>{transcription}</audio>"
                        }
                    ],
                    "audios": [audio_path]
                }
                
                # Write to JSONL file
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                processed += 1
    
    print(f"\n✓ Successfully processed {processed} entries")
    if skipped > 0:
        print(f"✗ Skipped {skipped} entries")
    print(f"✓ Output saved to: {output_file}")


def main():
    # Configuration
    csv_file = "/home/user/voice/anand/datasets/Hindi_male_mono/Hindi_male_mono.txt"  # Change this to your CSV file path
    audio_dir = "/home/user/voice/anand/datasets/Hindi_male_mono/Hindi_male_audio"
    output_file = "output.jsonl"
    
    # Verify CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print("Please update the 'csv_file' variable with the correct path.")
        return
    
    # Verify audio directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        print("Please update the 'audio_dir' variable with the correct path.")
        return
    
    print(f"Reading CSV file: {csv_file}")
    print(f"Looking for audio files in: {audio_dir}")
    print(f"Output will be saved to: {output_file}")
    print("-" * 60)
    
    create_jsonl_from_csv(csv_file, audio_dir, output_file)


if __name__ == "__main__":
    main()