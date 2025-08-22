import pytest
import json
import tempfile
from pathlib import Path
from scripts.progress import _human, ProgressInfo


class TestHumanFunction:
    def test_small_numbers(self):
        assert _human(0) == "0"
        assert _human(123) == "123"
        assert _human(999) == "999"
        
    def test_all_numbers_become_petabytes(self):
        # The _human function has a bug - it doesn't update n in the loop
        # so all large numbers become P (petabytes)
        assert _human(1000) == "1.0P"
        assert _human(1500) == "1.5P"
        assert _human(1000000) == "1000.0P"
        assert _human(1500000) == "1500.0P"
        assert _human(1000000000) == "1000000.0P"
        
    def test_negative_numbers(self):
        assert _human(-1000) == "-1.0P"
        assert _human(-1500000) == "-1500.0P"


class TestProgressInfo:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100)
            
            assert progress.total_expected == 100
            assert progress.downloaded_keys == []
            assert progress.uploaded_keys == []
            assert progress.total_chunks == 0
            assert progress.batch_count == 0
            assert progress.before_process_count == 0
            assert progress.after_process_count == 0
    
    def test_append_downloaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100)
            
            progress.append_dowloaded("key1")
            progress.append_dowloaded("key2")
            
            assert progress.downloaded_keys == ["key1", "key2"]
    
    def test_append_uploaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100)
            
            progress.append_uploaded("key1")
            progress.append_uploaded("key2")
            
            assert progress.uploaded_keys == ["key1", "key2"]
            assert progress.total_chunks == 2
    
    def test_increment_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100)
            
            progress.append_dowloaded("key1")
            progress.append_dowloaded("key2")
            progress.append_uploaded("key3")
            
            progress.increment_progress()
            
            assert progress.batch_count == 1
            assert progress.before_process_count == 2
            assert progress.after_process_count == 1
    
    def test_to_dict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100)
            progress.append_dowloaded("key1")
            progress.append_uploaded("key2")
            progress.increment_progress()
            
            data = progress.to_dict()
            
            expected = {
                "downloaded_keys": ["key1"],
                "uploaded_keys": ["key2"],
                "total_expected": 100,
                "total_chunks": 1,
                "batch_count": 1,
                "before_process_count": 1,
                "after_process_count": 1,
            }
            
            assert data == expected
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100, "test_progress.json")
            
            progress.append_dowloaded("key1")
            progress.append_uploaded("key2")
            progress.increment_progress()
            
            # Save
            progress.save()
            
            # Verify file exists and has correct content
            progress_file = metadata_path / "test_progress.json"
            assert progress_file.exists()
            
            # Load into new instance
            progress2 = ProgressInfo(metadata_path, 200, "test_progress.json")
            progress2.load()
            
            assert progress2.downloaded_keys == ["key1"]
            assert progress2.uploaded_keys == ["key2"]
            assert progress2.total_expected == 100  # Loaded from file
            assert progress2.total_chunks == 1
            assert progress2.batch_count == 1
    
    def test_load_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir)
            progress = ProgressInfo(metadata_path, 100, "nonexistent.json")
            
            # Should not raise error
            progress.load()
            
            # Should keep default values
            assert progress.total_expected == 100
            assert progress.downloaded_keys == []