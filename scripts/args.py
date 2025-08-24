from abc import ABC, abstractmethod

from logger import PrettyLogger, LogLevel
import argparse
from pathlib import Path
import constants as const


class Args(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def validate(self, logger: PrettyLogger) -> bool:
        pass

class PreprocessArgs(Args):

    def __init__(self):
        self.args = self.setup_args()

    def setup_args(self):
        """Setup command line arguments"""
        parser = argparse.ArgumentParser(
            description="Process audio files for TTS training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # I/O paths
        parser.add_argument(
            "--in_path", type=Path, default=const.DEFAULT_IN_PATH,
            help="Input path for audio files"
        )
        parser.add_argument(
            "--out_path", type=Path, default=const.DEFAULT_OUT_PATH,
            help="Output path for processed files"
        )
        parser.add_argument(
            "--metadata_path", type=Path, default=const.DEFAULT_METADATA_PATH,
            help="Path to save metadata files"
        )
        
        # Processing parameters
        parser.add_argument(
            "--lan", type=str, default=const.DEFAULT_LANGUAGE,
            help="Language for transcription and processing"
        )
        parser.add_argument(
            "--csv_filename", type=Path, default=const.DEFAULT_CSV_FILENAME,
            help="Path to save the metadata CSV file"
        )
        parser.add_argument(
            "--whisper_batch_size", type=int, default=const.WHISPER_BATCH,
            help="Batch size for Whisper processing"
        )
        parser.add_argument(
            "--whisper_size", type=str, default=const.WHISPER_SIZE,
            help="Model size for Whisper (tiny, base, small, medium, large)"
        )
        parser.add_argument(
            "--target_sr", type=int, default=const.DEFAULT_TARGET_SR,
            help="Target sample rate for processed audio files"
        )
        
        # S3 configuration
        parser.add_argument(
            "--s3_bucket", type=str, default=const.DEFAULT_S3_BUCKET,
            help="S3 bucket name"
        )
        parser.add_argument(
            "--s3_processed_prefix", type=str, default=const.DEFAULT_S3_PROCESSED_PREFIX,
            help="S3 prefix for processed files"
        )
        parser.add_argument(
            "--s3_batch_size", type=int, default=const.DEFAULT_S3_BATCH_SIZE,
            help="Number of files to process in each S3 batch"
        )
        
        # Options
        parser.add_argument(
            "--only_meta", action="store_true",
            help="Only upload metadata, skip processing"
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true",
            help="Enable verbose logging (debug level)"
        )
        parser.add_argument(
            "--dry_run", action="store_true",
            help="Show what would be processed without actually doing it"
        )
        parser.add_argument(
            "--live_dashboard", action="store_true",
            help="Enable live status dashboard during processing"
        )
        
        return parser.parse_args()
    
    
    def validate_args(self, logger: PrettyLogger) -> bool:
        """Validate command line arguments"""
        with logger.step("config_validation", "Validate configuration") as step:
            errors = []
            warnings = []
            
            step.log("Checking Whisper configuration...")
            
            # Check whisper model size
            valid_sizes = ["tiny", "base", "small", "medium", "large"]
            if self.args.whisper_size not in valid_sizes:
                errors.append(f"Invalid whisper_size: {self.args.whisper_size}. Must be one of {valid_sizes}")
            
            # Check target sample rate
            if self.args.target_sr <= 0:
                errors.append(f"Invalid target_sr: {self.args.target_sr}. Must be positive")
            elif self.args.target_sr < 8000:
                warnings.append(f"Very low target_sr (self.{self.args.target_sr}Hz) may cause quality issues")
            elif self.args.target_sr < 16000:
                warnings.append(f"Low target_sr ({self.args.target_sr}Hz) may affect model performance")
            
            step.log("Checking batch size configuration...")
            
            # Check batch sizes
            if self.args.whisper_batch_size <= 0:
                errors.append(f"Invalid whisper_batch_size: {self.args.whisper_batch_size}")
            elif self.args.whisper_batch_size > 32:
                warnings.append(f"Large whisper_batch_size ({self.args.whisper_batch_size}) may cause GPU OOM")
                
            if self.args.s3_batch_size <= 0:
                errors.append(f"Invalid s3_batch_size: {self.args.s3_batch_size}")
            elif self.args.s3_batch_size > 1000:
                warnings.append(f"Very large s3_batch_size ({self.args.s3_batch_size}) may be inefficient")
            
            step.log("Checking paths and S3 configuration...")
            
            # Check if S3 bucket name looks valid
            if not self.args.s3_bucket or ' ' in self.args.s3_bucket:
                warnings.append(f"S3 bucket name '{self.args.s3_bucket}' may be invalid")
            
            # Log all findings
            for warning in warnings:
                step.log(warning, LogLevel.WARN)
            
            for error in errors:
                step.log(error, LogLevel.ERROR)
            
            # Update step with validation results
            step.update({
                "validation_errors": len(errors),
                "validation_warnings": len(warnings),
                "whisper_model": self.args.whisper_size,
                "language": self.args.lan,
                "target_sr": self.args.target_sr,
                "s3_bucket": self.args.s3_bucket,
                "whisper_batch_size": self.args.whisper_batch_size,
                "s3_batch_size": self.args.s3_batch_size
            })
            
            if errors:
                step.log(f"Configuration validation failed with {len(errors)} errors")
                return False
            elif warnings:
                step.log(f"Configuration valid with {len(warnings)} warnings")
            else:
                step.log("Configuration validation passed")
            
            return True
    
    
    
    
