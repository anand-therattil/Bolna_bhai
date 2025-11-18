import pandas as pd
import soundfile as sf
import os
import json
from pathlib import Path
import io
from tqdm import tqdm
import concurrent.futures
import traceback

def process_sample(sample_data):
    """
    Verarbeitet ein einzelnes Audiosample: dekodiert Audio, schreibt WAV- und TXT-Dateien
    und gibt den Metadateneintrag und die Dauer zurück.
    Diese Funktion ist so konzipiert, dass sie in einem separaten Thread ausgeführt wird.

    Args:
        sample_data (dict): Ein Wörterbuch mit den Daten für ein einzelnes Sample,
                            einschließlich 'index', 'row', 'output_path', 'dataset_name'.

    Returns:
        tuple: Ein Tupel, das den Metadateneintrag (dict) und die Dauer des Audios (float) enthält.
               Gibt (None, None) zurück, wenn ein Fehler auftritt.
    """
    try:
        index = sample_data['index']
        row = sample_data['row']
        output_path = sample_data['output_path']
        dataset_name = sample_data['dataset_name']
        file_path_name = sample_data['file_path_name'] # Originaler Parquet-Dateiname für besseres Debugging

        audio_info = row['audio']
        transcription = row['transcription']

        # Audio-Bytes dekodieren
        audio_bytes = audio_info['bytes']
        audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        
        duration = len(audio_data) / sampling_rate

        # Eindeutige Dateinamen generieren
        file_id = f"{dataset_name}_speaker_{index:06d}"
        wav_filename = f"{file_id}.wav"
        txt_filename = f"{file_id}.txt"
        
        wav_filepath = output_path / wav_filename
        txt_filepath = output_path / txt_filename

        # E/A-Operationen: Schreiben der WAV- und TXT-Dateien
        sf.write(wav_filepath, audio_data, sampling_rate)
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(transcription)

        # Metadateneintrag für dieses Sample erstellen
        sample_entry = {
            "id": file_id,
            "audio_file": str(wav_filename),
            "transcript_file": str(txt_filename),
            "duration": round(duration, 2),
            "speaker_id": f"{dataset_name}_speaker",
            "speaker_name": dataset_name.capitalize(),
            "scene": "unknown",
            "emotion": "neutral",
            "language": "vi",
            "gender": "unknown",
            "quality_score": 1.0,
            "original_audio_path": audio_info.get('path', f"from_{file_path_name}_row_{row.name}"),
            "user_instruction": "<audio> /translate",
            "task_type": "audio_generation"
        }
        
        return sample_entry, duration

    except Exception as e:
        # Fehlerinformationen für die spätere Überprüfung ausgeben
        print(f"\nFehler bei der Verarbeitung von Sample {index}: {e}")
        traceback.print_exc()
        return None, None

def process_parquet_files(input_dir, output_dir, dataset_name="huo", max_workers=None):
    """
    Verarbeitet Parquet-Dateien, extrahiert Audio und Transkriptionen und generiert 
    entsprechende WAV-, TXT-Dateien und eine metadata.json mithilfe von Multithreading
    für E/A-Operationen.
    """
    if max_workers is None:
        # Eine gute Standardeinstellung ist die Anzahl der CPUs + 4, was bei E/A-gebundenen Aufgaben gut funktioniert.
        # os.cpu_count() kann None zurückgeben, daher der Fallback auf 4.
        max_workers = (os.cpu_count() or 4) + 4

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_samples = []
    total_duration = 0.0
    processed_count = 0
    
    parquet_files = sorted(list(input_path.glob("train-*.parquet")))
    if not parquet_files:
        print(f"Keine Parquet-Dateien im Verzeichnis gefunden: {input_path}")
        return

    # --- Schritt 1: Gesamtzahl der Samples für die Fortschrittsanzeige berechnen ---
    print("Berechne Gesamtzahl der Samples für die Fortschrittsanzeige...")
    total_samples = 0
    for file_path in tqdm(parquet_files, desc="Dateien scannen"):
        try:
            df = pd.read_parquet(file_path, columns=['audio'])
            total_samples += len(df)
        except Exception as e:
            print(f"Fehler beim Scannen der Datei {file_path}: {e}")
    
    if total_samples == 0:
        print("Keine Samples in den Parquet-Dateien gefunden.")
        return
        
    print(f"{total_samples} Samples gefunden, starte Verarbeitung mit bis zu {max_workers} Threads...")

    # --- Schritt 2: Verarbeitung der Dateien mit einem ThreadPoolExecutor ---
    global_sample_index = 0
    with tqdm(total=total_samples, desc="Samples verarbeiten", unit="sample") as pbar:
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                
                tasks = []
                # Erstellen Sie einen ThreadPoolExecutor, um Aufgaben für die aktuelle Parquet-Datei zu verwalten
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Senden Sie eine Aufgabe für jede Zeile im DataFrame
                    for _, row in df.iterrows():
                        sample_data = {
                            'index': global_sample_index,
                            'row': row,
                            'output_path': output_path,
                            'dataset_name': dataset_name,
                            'file_path_name': file_path.name,
                        }
                        tasks.append(executor.submit(process_sample, sample_data))
                        global_sample_index += 1
                    
                    # Verarbeiten Sie die Ergebnisse, sobald sie eintreffen
                    for future in concurrent.futures.as_completed(tasks):
                        result, duration = future.result()
                        if result and duration is not None:
                            metadata_samples.append(result)
                            total_duration += duration
                            processed_count += 1
                        # Aktualisieren Sie die Fortschrittsanzeige für jedes abgeschlossene Sample
                        pbar.update(1)

            except Exception as e:
                pbar.write(f"\nFehler bei der Verarbeitung der Datei {file_path}: {e}")
                traceback.print_exc()

    # --- Schritt 3: Endgültige Metadaten erstellen und speichern ---
    print("\nErstelle endgültige Metadatendatei...")
    
    avg_duration = (total_duration / processed_count) if processed_count > 0 else 0.0

    metadata = {
        "dataset_info": {
            "total_samples": processed_count,
            "speakers": [f"{dataset_name}_speaker"],
            "languages": ["vi"],
            "total_duration": round(total_duration, 2),
            "avg_duration": round(avg_duration, 2),
            "created_from": [str(input_path.resolve())]
        },
        "samples": metadata_samples
    }

    metadata_filepath = output_path / "metadata.json"
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nVerarbeitung abgeschlossen! Dateien wurden gespeichert unter: {output_path}")
    print(f"Insgesamt {processed_count} von {total_samples} Samples erfolgreich verarbeitet.")


if __name__ == '__main__':
    # ***** Bitte passen Sie die folgenden Variablen an Ihre Pfade an *****
    INPUT_DATA_DIR = '/root/datasets/doof-ferb/vlsp2020_vinai_100h/data'
    OUTPUT_DATA_DIR = '/root/code/higgs-audio-main/higgs_training_data_mini_vn'
    DATASET_NAME = 'vn'
    
    # Optional: Legen Sie die maximale Anzahl von Worker-Threads fest.
    # Wenn Sie None angeben, wird ein sinnvoller Standardwert berechnet.
    MAX_WORKERS = 16 

    input_dir_expanded = os.path.expanduser(INPUT_DATA_DIR)
    output_dir_expanded = os.path.expanduser(OUTPUT_DATA_DIR)
    
    process_parquet_files(input_dir_expanded, output_dir_expanded, DATASET_NAME, max_workers=MAX_WORKERS)
