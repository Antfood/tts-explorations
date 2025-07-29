from scripts import preprocess
from pathlib import Path

import csv

DEFAULT_OUT_PATH = Path("./processed")

in_path = Path("./data")
csv_path = DEFAULT_OUT_PATH / "metadata.csv"
preprocess.set_language("pt")

def preproces(in_path: Path, out_path: Path):
    out_path = DEFAULT_OUT_PATH
    wave_paths = list(in_path.glob("*.wav"))

    for audio_path in wave_paths:
        audio, transcription = preprocess.transcribe(audio_path)

        aligned_results = preprocess.align(audio, transcription)
        out_path.mkdir(parents=True, exist_ok=True)

        yield preprocess.split_audio(audio_path, out_path, aligned_results)


with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    headers = preprocess.ProcessedChunk.headers()
    writer.writerow(headers)

    for chunks in preproces(in_path, DEFAULT_OUT_PATH):
        for chunk in chunks:
            writer.writerow(chunk.to_list())
