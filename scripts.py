from scripts import preprocess
from pathlib import Path
import argparse

import csv

DEFAULT_OUT_PATH = Path("./processed")
DEFAULT_IN_PATH = Path("./data")
DEFAULT_LANGUAGE = "pt"
DEFAULT_CSV_PATH = DEFAULT_OUT_PATH / "metadata.csv"

def preproces(in_path: Path, out_path: Path):
    out_path = DEFAULT_OUT_PATH
    wave_paths = list(in_path.glob("*.wav"))

    for audio_path in wave_paths:
        audio, transcription = preprocess.transcribe(audio_path)

        aligned_results = preprocess.align(audio, transcription)
        out_path.mkdir(parents=True, exist_ok=True)

        yield preprocess.split_audio(audio_path, out_path, aligned_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument("--in_path", type=Path, default=DEFAULT_OUT_PATH, help="Input path for audio files")
    parser.add_argument("--out_path", type=Path, default=DEFAULT_OUT_PATH, help="Output path for processed files")
    parser.add_argument("--lan", type=str, default=DEFAULT_LANGUAGE, help="Language for transcription and processing")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH, help="Path to save the metadata CSV file")

    args = parser.parse_args()
    preprocess.set_language(args.lan)
    

    with open(args.csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        headers = preprocess.ProcessedChunk.headers()
        writer.writerow(headers)
    
        for chunks in preproces(args.in_path, args.out_path):
            for chunk in chunks:
                writer.writerow(chunk.to_list())
    

