from scripts.preprocessor import Preprocessor, ProcessedChunk
from scripts.s3_batcher import S3Batcher
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
        "--metadata_path",
        type=Path,
        default=const.DEFAULT_METADATA_PATH,
        help="Path to save metadata files",
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

    parser.add_argument(
        "--whisper_batch_size",
        type=int,
        default=const.WHISPER_BATCH,
        help="Batch size for Whisper processing",
    )

    parser.add_argument(
        "--whisper_size",
        type=str,
        default=const.WHISPER_SIZE,
        help="Model size for Whisper (tiny, base, small, medium, large)",
    )

    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=const.DEFAULT_S3_BUCKET,
        help="S3 bucket name",
    )

    parser.add_argument(
        "--s3_processed_prefix",
        type=str,
        default=const.DEFAULT_S3_PROCESSED_PREFIX,
        help="S3 prefix for processed files",
    )

    parser.add_argument(
        "--s3_batch_size",
        type=int,
        default=const.DEFAULT_S3_BATCH_SIZE,
        help="Number of files to process in each S3 batch",
    )

    parser.add_argument(
        "--only_meta",
        default=False,
        type=bool,
    )

    args = parser.parse_args()

    proc = Preprocessor(
        in_path=args.in_path,
        out_path=args.out_path,
        model_size=args.whisper_size,
        compute_type="float32",
        language=args.lan,
        metadata_path=args.metadata_path,
        batch_size=args.whisper_batch_size,
    )

    batcher = S3Batcher(
        download_to=args.in_path,
        upload_from=args.out_path,
        bucket=args.s3_bucket,
        metadata_path=args.metadata_path,
        processed_prefix=args.s3_processed_prefix,
        batch_size=args.s3_batch_size,
    )

    # Make sure directories exist
    args.out_path.mkdir(parents=True, exist_ok=True)
    args.in_path.mkdir(parents=True, exist_ok=True)
    args.metadata_path.mkdir(parents=True, exist_ok=True)

    csv_path = args.metadata_path / args.csv_filename

    if args.only_meta:
        batcher.upload_metadata()
        exit(0)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        headers = ProcessedChunk.headers()
        writer.writerow(headers)

        while batcher.has_next():
            batcher.next_batch()  # grabs batch from S3

            # procress files
            for chunks in proc.preprocess():
                for chunk in chunks:
                    writer.writerow(chunk.to_list())

            batcher.upload()  # upload processed files to S3
            # clean up processed files

    batcher.upload_metadata()  # upload metadata to S3
