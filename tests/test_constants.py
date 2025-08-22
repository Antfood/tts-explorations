import pytest
from pathlib import Path
from scripts import constants


class TestConstants:
    def test_path_constants_are_paths(self):
        assert isinstance(constants.DEFAULT_OUT_PATH, Path)
        assert isinstance(constants.DEFAULT_IN_PATH, Path)
        assert isinstance(constants.DEFAULT_METADATA_PATH, Path)
    
    def test_path_values(self):
        assert constants.DEFAULT_OUT_PATH == Path("./processed")
        assert constants.DEFAULT_IN_PATH == Path("./data")
        assert constants.DEFAULT_METADATA_PATH == Path("./metadata")
    
    def test_string_constants(self):
        assert constants.DEFAULT_LANGUAGE == "pt"
        assert constants.DEFAULT_CSV_FILENAME == "metadata.csv"
        assert constants.DEFAULT_S3_PROCESSED_PREFIX == "processed"
        assert constants.DEFAULT_S3_BUCKET == "mi-lou-vo"
    
    def test_numeric_constants(self):
        assert isinstance(constants.WHISPER_BATCH, int)
        assert constants.WHISPER_BATCH == 8
        assert constants.DEFAULT_S3_BATCH_SIZE == 50
    
    def test_whisper_model_size(self):
        assert constants.WHISPER_SIZE == "small"
        # Verify it's a valid model size (common Whisper sizes)
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        assert constants.WHISPER_SIZE in valid_sizes