from scripts import preprocessor
from scripts import constants as const
from pathlib import Path
import argparse

import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument(
        "--in_path",
        type=Path,
        default=const.DEFAULT_IN_PATH,
        help="Input path for audio files",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=const.DEFAULT_OUT_PATH,
        help="Output path for processed files",
    )
    parser.add_argument(
        "--lan",
        type=str,
        default=const.DEFAULT_LANGUAGE,
        help="Language for transcription and processing",
    )
    parser.add_argument(
        "--csv_filename",
        type=Path,
        default=const.DEFAULT_CSV_FILENAME,
        help="Path to save the metadata CSV file",
    )

    args = parser.parse_args()
    preprocessor.set_language(args.lan)
    csv_path = args.out_path / args.csv_filename
    args.out_path.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        headers = preprocessor.ProcessedChunk.headers()
        writer.writerow(headers)

        for chunks in preprocessor.preprocess(args.in_path, args.out_path):
            print( f":: Writing {len(chunks)} chunks to CSV for file {chunks[0].original_audio_path}")
            for chunk in chunks:
                writer.writerow(chunk.to_list())
