from scripts.preprocessor import Preprocessor, ProcessedChunk
from scripts.s3_batcher import S3Batcher
from scripts.logger import PrettyLogger, LogLevel
from scripts import constants as const
from pathlib import Path
import argparse
import csv
import sys
import traceback
import time


def setup_args():
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


def validate_args(args, logger: PrettyLogger):
    """Validate command line arguments"""
    with logger.step("config_validation", "Validate configuration") as step:
        errors = []
        warnings = []
        
        step.log("Checking Whisper configuration...")
        
        # Check whisper model size
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if args.whisper_size not in valid_sizes:
            errors.append(f"Invalid whisper_size: {args.whisper_size}. Must be one of {valid_sizes}")
        
        # Check target sample rate
        if args.target_sr <= 0:
            errors.append(f"Invalid target_sr: {args.target_sr}. Must be positive")
        elif args.target_sr < 8000:
            warnings.append(f"Very low target_sr ({args.target_sr}Hz) may cause quality issues")
        elif args.target_sr < 16000:
            warnings.append(f"Low target_sr ({args.target_sr}Hz) may affect model performance")
        
        step.log("Checking batch size configuration...")
        
        # Check batch sizes
        if args.whisper_batch_size <= 0:
            errors.append(f"Invalid whisper_batch_size: {args.whisper_batch_size}")
        elif args.whisper_batch_size > 32:
            warnings.append(f"Large whisper_batch_size ({args.whisper_batch_size}) may cause GPU OOM")
            
        if args.s3_batch_size <= 0:
            errors.append(f"Invalid s3_batch_size: {args.s3_batch_size}")
        elif args.s3_batch_size > 1000:
            warnings.append(f"Very large s3_batch_size ({args.s3_batch_size}) may be inefficient")
        
        step.log("Checking paths and S3 configuration...")
        
        # Check if S3 bucket name looks valid
        if not args.s3_bucket or ' ' in args.s3_bucket:
            warnings.append(f"S3 bucket name '{args.s3_bucket}' may be invalid")
        
        # Log all findings
        for warning in warnings:
            step.log(warning, LogLevel.WARN)
        
        for error in errors:
            step.log(error, LogLevel.ERROR)
        
        # Update step with validation results
        step.update({
            "validation_errors": len(errors),
            "validation_warnings": len(warnings),
            "whisper_model": args.whisper_size,
            "language": args.lan,
            "target_sr": args.target_sr,
            "s3_bucket": args.s3_bucket,
            "whisper_batch_size": args.whisper_batch_size,
            "s3_batch_size": args.s3_batch_size
        })
        
        if errors:
            step.log(f"Configuration validation failed with {len(errors)} errors")
            return False
        elif warnings:
            step.log(f"Configuration valid with {len(warnings)} warnings")
        else:
            step.log("Configuration validation passed")
        
        return True


def setup_directories(args, logger: PrettyLogger):
    """Ensure all required directories exist"""
    with logger.step("directory_setup", "Setup directories") as step:
        directories = [args.out_path, args.in_path, args.metadata_path]
        
        created_dirs = []
        existing_dirs = []
        
        for directory in directories:
            if directory.exists():
                existing_dirs.append(directory)
                step.log(f"Directory exists: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)
                step.log(f"Created directory: {directory}")
        
        step.update({
            "created_directories": [str(d) for d in created_dirs],
            "existing_directories": [str(d) for d in existing_dirs],
            "total_directories": len(directories),
            "metadata_csv_path": str(args.metadata_path / args.csv_filename)
        })
        
        if created_dirs:
            step.log(f"Created {len(created_dirs)} new directories")
        step.log("Directory setup complete")


def handle_metadata_only(args, logger: PrettyLogger):
    """Handle metadata-only upload mode"""
    with logger.step("metadata_only_mode", "Metadata upload mode") as step:
        step.log("Running in metadata-only mode")
        step.update({
            "mode": "metadata_only",
            "s3_bucket": args.s3_bucket,
            "metadata_path": str(args.metadata_path)
        })
        
        batcher = S3Batcher(
            download_to=args.in_path,
            upload_from=args.out_path,
            bucket=args.s3_bucket,
            metadata_path=args.metadata_path,
            processed_prefix=args.s3_processed_prefix,
            batch_size=args.s3_batch_size,
            logger=logger,
        )
        
        batcher.upload_metadata()
        step.log("Metadata upload completed")


def initialize_components(args, logger: PrettyLogger):
    """Initialize processing components"""
    with logger.step("component_initialization", "Initialize processing components") as step:
        
        step.log("Creating preprocessor...")
        try:
            proc = Preprocessor(
                in_path=args.in_path,
                out_path=args.out_path,
                model_size=args.whisper_size,
                compute_type="float32",
                language=args.lan,
                metadata_path=args.metadata_path,
                batch_size=args.whisper_batch_size,
                target_sr=args.target_sr,
                logger=logger,
            )
            step.log("Preprocessor initialized successfully")
        except Exception as e:
            step.log(f"Failed to initialize preprocessor: {e}", LogLevel.ERROR)
            raise
        
        step.log("Creating S3 batcher...")
        try:
            batcher = S3Batcher(
                download_to=args.in_path,
                upload_from=args.out_path,
                bucket=args.s3_bucket,
                metadata_path=args.metadata_path,
                processed_prefix=args.s3_processed_prefix,
                batch_size=args.s3_batch_size,
                logger=logger,
            )
            step.log("S3 batcher initialized successfully")
        except Exception as e:
            step.log(f"Failed to initialize S3 batcher: {e}", LogLevel.ERROR)
            raise
        
        step.update({
            "preprocessor_device": proc.device,
            "whisper_model": args.whisper_size,
            "s3_bucket": args.s3_bucket,
            "total_s3_files": batcher.total,
            "target_language": args.lan
        })
        
        return proc, batcher


def run_processing_pipeline(args, proc, batcher, logger: PrettyLogger):
    """Run the main processing pipeline"""
    
    # Initialize output files
    proc.init_files()
    
    csv_path = args.metadata_path / args.csv_filename
    total_processed_chunks = 0
    batch_count = 0
    failed_batches = 0
    
    pipeline_start_time = time.time()
    
    with logger.step("pipeline_execution", "Execute processing pipeline") as pipeline_step:
        
        pipeline_step.log("Starting batch processing loop...")
        pipeline_step.update({
            "csv_output": str(csv_path),
            "s3_bucket": args.s3_bucket,
            "total_files_in_bucket": batcher.total
        })
        
        # Open CSV file for writing
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            headers = ProcessedChunk.headers()
            writer.writerow(headers)
            
            # Main processing loop
            while batcher.has_next():
                batch_count += 1
                batch_start_time = time.time()
                
                batch_step_id = f"batch_{batch_count}"
                
                try:
                    with logger.step(batch_step_id, f"Process batch #{batch_count}") as batch_step:
                        batch_step.update({
                            "batch_number": batch_count,
                            "total_chunks_so_far": total_processed_chunks
                        })
                        
                        # Download batch from S3
                        batch_step.log("Downloading files from S3...")
                        batcher.next_batch()
                        
                        if args.dry_run:
                            batch_step.log("DRY RUN: Skipping actual processing")
                            batch_step.update({"mode": "dry_run"})
                            continue
                        
                        # Process downloaded files
                        batch_step.log("Starting audio processing...")
                        batch_chunks = 0
                        processed_files = 0
                        
                        for chunks in proc.preprocess():
                            if chunks:  # Only count non-empty chunk lists
                                processed_files += 1
                                for chunk in chunks:
                                    writer.writerow(chunk.to_list())
                                    batch_chunks += 1
                        
                        total_processed_chunks += batch_chunks
                        batch_duration = time.time() - batch_start_time
                        
                        batch_step.update({
                            "files_processed_this_batch": processed_files,
                            "chunks_created_this_batch": batch_chunks,
                            "total_chunks_created": total_processed_chunks,
                            "batch_duration_seconds": round(batch_duration, 1),
                            "chunks_per_minute": round((batch_chunks / batch_duration) * 60, 1) if batch_duration > 0 else 0
                        })
                        
                        # Upload processed files
                        if batch_chunks > 0:
                            batch_step.log("Uploading processed files...")
                            batcher.upload()
                            batch_step.log(f"Batch #{batch_count} complete: {batch_chunks} chunks created")
                        else:
                            batch_step.log("No chunks created in this batch", LogLevel.WARN)
                        
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"Batch #{batch_count} failed: {e}")
                    if args.verbose:
                        logger.error(traceback.format_exc())
                    # Continue with next batch instead of crashing
                    continue
        
        # Update pipeline summary
        total_pipeline_time = time.time() - pipeline_start_time
        pipeline_step.update({
            "total_batches_processed": batch_count,
            "successful_batches": batch_count - failed_batches,
            "failed_batches": failed_batches,
            "total_chunks_created": total_processed_chunks,
            "total_pipeline_duration_seconds": round(total_pipeline_time, 1),
            "average_chunks_per_batch": round(total_processed_chunks / batch_count, 1) if batch_count > 0 else 0,
            "chunks_per_hour": round((total_processed_chunks / total_pipeline_time) * 3600, 1) if total_pipeline_time > 0 else 0
        })
        
        if failed_batches > 0:
            pipeline_step.log(f"Pipeline completed with {failed_batches} failed batches", LogLevel.WARN)
        else:
            pipeline_step.log("All batches processed successfully")
    
    return total_processed_chunks, batch_count, failed_batches


def main():
    """Main processing pipeline"""
    # Initialize basic variables
    start_time = time.time()
    args = None
    logger = None
    
    try:
        args = setup_args()
        logger = PrettyLogger()
        
        # Header
        with logger.panel_section("🎤 TTS Audio Preprocessing Pipeline", "cyan"):
            logger.info("Starting automated audio preprocessing for TTS training")
            logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.verbose:
            logger.info("Verbose logging enabled")
        
        if not validate_args(args, logger):
            logger.error("❌ Configuration validation failed")
            sys.exit(1)
        
        # Setup directories
        setup_directories(args, logger)
        
        # Handle special modes
        if args.only_meta:
            handle_metadata_only(args, logger)
            logger.success("✅ Metadata upload completed")
            return
        
        if args.dry_run:
            logger.info("🔍 Running in DRY RUN mode - no actual processing")
        
        # Initialize components
        proc, batcher = initialize_components(args, logger)
        
        # Run main processing pipeline
        processing_context = logger.live_status_panel if args.live_dashboard else logger.panel_section
        context_args = ("📊 Live Processing Dashboard",) if args.live_dashboard else ("🚀 Processing Pipeline", "blue")
        
        with processing_context(*context_args):
            total_chunks, total_batches, failed_batches = run_processing_pipeline(args, proc, batcher, logger)
        
        # Final metadata upload
        with logger.step("final_metadata_upload", "Upload final metadata") as step:
            step.log("Uploading final metadata files to S3...")
            batcher.upload_metadata()
            
            step.update({
                "csv_file": str(args.metadata_path / args.csv_filename),
                "total_chunks_processed": total_chunks,
                "total_batches_processed": total_batches,
                "failed_batches": failed_batches
            })
            
            step.log("Final metadata upload completed")
        
        # Success summary
        total_time = time.time() - start_time
        
        with logger.panel_section("🎉 Pipeline Complete", "green"):
            if failed_batches == 0:
                logger.success("✅ Pipeline completed successfully!")
            else:
                logger.success(f"✅ Pipeline completed with {failed_batches} failed batches")
            
            logger.success(f"📊 Results: {total_chunks:,} audio chunks from {total_batches} batches")
            logger.success(f"⏱️  Total time: {total_time/60:.1f} minutes")
            
            if total_chunks > 0:
                logger.success(f"🚀 Processing rate: {(total_chunks/total_time)*60:.1f} chunks/minute")
        
    except KeyboardInterrupt:
        if logger:
            logger.error("❌ Pipeline interrupted by user (Ctrl+C)")
        else:
            print("❌ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
        
    except SystemExit as e:
        # Handle argparse --help or other legitimate exits
        # Don't show error messages for help or version
        if e.code == 0:  # Normal exit (like --help)
            sys.exit(0)
        else:
            if logger:
                logger.error(f"❌ Pipeline exited with code {e.code}")
            sys.exit(e.code)
        
    except Exception as e:
        total_time = time.time() - start_time
        
        if logger:
            logger.error(f"❌ Pipeline failed after {total_time/60:.1f} minutes")
            logger.error(f"Error: {str(e)}")
            
            # Only show verbose traceback if args is available and verbose is True
            if args and getattr(args, 'verbose', False):
                logger.error("Full traceback:")
                for line in traceback.format_exc().splitlines():
                    logger.error(line)
            elif not args:
                # If args failed to parse, always show traceback for debugging
                logger.error("Error occurred during startup - showing full traceback:")
                for line in traceback.format_exc().splitlines():
                    logger.error(line)
        else:
            # Fallback to basic print if logger not initialized
            print(f"❌ Pipeline failed after {total_time/60:.1f} minutes")
            print(f"Error: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
        
        sys.exit(1)
    
    finally:
        # Only print summary if logger was initialized (not for --help, etc.)
        if logger:
            logger.print_summary()


if __name__ == "__main__":
    main()
