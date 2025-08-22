from pathlib import Path

DEFAULT_OUT_PATH = Path("./processed")
DEFAULT_IN_PATH = Path("./data")
DEFAULT_METADATA_PATH = Path("./metadata")
DEFAULT_LANGUAGE = "pt"
DEFAULT_CSV_FILENAME = "metadata.csv"
WHISPER_BATCH = 8
WHISPER_SIZE = "small"
DEFAULT_S3_PROCESSED_PREFIX = "processed"
DEFAULT_S3_BUCKET = "mi-lou-vo"
DEFAULT_S3_BATCH_SIZE = 50


