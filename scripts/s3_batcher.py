import boto3
import os
from pathlib import Path
import json
from typing import List, Optional
import shutil

from .progress import ProgressInfo
from .logger import PrettyLogger, LogLevel


class BatcherState:
    def __init__(self, metadata_path: Path, name: str = "batcher_state.json"):
        self.metadata_path = metadata_path
        self.state_file = metadata_path / name
        self.token = None
        self.completed: bool = False
        self.load()

    def load(self):
        """Load the state of the batcher from a file."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.token = data.get("token", None)

    def save(self):
        """Save the state of the batcher to a file."""
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self):
        """Convert the state to a dictionary for saving."""
        return {
            "token": self.token,
            "completed": self.completed,
        }

    def token_from_response(self, resp: dict):
        self.token = resp.get("NextContinuationToken", None)
        if not self.token:
            self.completed = True

    def has_token(self) -> bool:
        """Check if this is the first batch."""
        return self.token is not None


class S3Batcher:
    def __init__(
        self,
        download_to: Path,
        upload_from: Path,
        metadata_path: Path,
        processed_prefix: str,
        bucket: str,
        metadata_prefix: str = "metadata",
        batch_size: int = 25,
        include_ext: tuple[str, ...] = (".wav",),
        logger: Optional[PrettyLogger] = None,
    ):
        self.client = boto3.client("s3")
        self.bucket = bucket
        self.download_to = download_to
        self.upload_from = upload_from
        self.batch_size = batch_size
        self.state = BatcherState(metadata_path)
        self.include_ext = include_ext
        self.metadata_path = metadata_path
        self.processed_prefix = processed_prefix
        self.metadata_prefix = metadata_prefix
        self.count = 0
        self.ignore_prefixes = [f"{metadata_prefix}/", f"{processed_prefix}/"]
        self.logger = logger or PrettyLogger()

        # Initialize with bucket scanning
        with self.logger.step("s3_init", "Initialize S3 batcher") as step:
            step.log(f"Connecting to S3 bucket: {bucket}")
            step.update({
                "bucket": bucket,
                "batch_size": batch_size,
                "include_extensions": list(include_ext),
                "processed_prefix": processed_prefix
            })
            
            step.log("Scanning bucket for total file count...")
            self.total = self.counter()
            
            step.update({"total_files_found": self.total})
            
            if self.total == 0:
                step.log("No files found in bucket", LogLevel.WARN)
            else:
                step.log(f"Found {self.total:,} files to process")

        self.progress = ProgressInfo(metadata_path, self.total)
        self.progress.load()  # Load existing progress

    def has_next(self) -> bool:
        """Check if there are more files to process."""
        return not self.state.completed

    def next_batch(self):
        """Get the next batch of files to process from S3."""
        batch_num = self.progress.batch_count + 1
        
        with self.logger.step(f"batch_download_{batch_num}", f"Download batch #{batch_num}") as step:
            # Preparation
            step.log("Preparing for next batch...")
            self.count = 0
            self.clear_local_dirs()
            
            # Load progress from disk
            self.progress.load()
            
            step.update({
                "batch_number": batch_num,
                "target_batch_size": self.batch_size,
                "continuation_token": self.format_token(self.state.token),
                "completed_so_far": len(self.progress.downloaded_keys)
            })
            
            step.log(f"Starting batch #{batch_num} download...")
            
            # Main download loop
            while self.count < self.batch_size and not self.state.completed:
                try:
                    step.log("Fetching file listing from S3...")
                    params = self.to_params()
                    resp = self.client.list_objects_v2(**params)
                    
                    self.state.token_from_response(resp)
                    batch = resp.get("Contents", [])
                    
                    if not batch:
                        step.log("No more files available in S3")
                        self.empty_batch()
                        break
                    
                    step.log(f"Retrieved {len(batch)} file entries from S3 API")
                    
                    # Download files from this API response
                    downloaded = self.download_batch_files(batch, step)
                    
                    if downloaded == 0:
                        step.log("No valid files in this batch, continuing...")
                        continue
                        
                    # Check if we've reached our batch size or if this was a small response
                    if self.count >= self.batch_size or len(batch) < 1000:
                        break
                        
                except Exception as e:
                    step.log(f"Error during S3 operations: {e}", LogLevel.ERROR)
                    raise
            
            step.update({
                "files_downloaded": self.count,
                "batch_complete": self.state.completed,
                "next_token": self.format_token(self.state.token)
            })
            
            if self.count > 0:
                step.log(f"Batch download complete: {self.count} files ready for processing")
            else:
                step.log("Batch download complete: No files downloaded", LogLevel.WARN)

        # Save state after batch completion
        self.state.save()

    def download_batch_files(self, batch: List[dict], step_context) -> int:
        """Download files from a batch, return count of successfully downloaded files"""
        # Filter valid files first
        valid_files = []
        skipped_count = 0
        
        for obj in batch:
            key = obj["Key"]
            size_mb = obj.get("Size", 0) / 1024 / 1024
            
            # Apply filters
            if self.include_ext and not key.lower().endswith(self.include_ext):
                skipped_count += 1
                continue
                
            if any(key.startswith(p) for p in self.ignore_prefixes):
                skipped_count += 1
                continue
            
            valid_files.append((key, size_mb))
        
        if skipped_count > 0:
            step_context.log(f"Filtered out {skipped_count} files (extensions/prefixes)")
        
        if not valid_files:
            return 0
        
        # Limit to remaining batch size
        files_to_download = min(len(valid_files), self.batch_size - self.count)
        total_size = sum(size for _, size in valid_files[:files_to_download])
        
        step_context.log(f"Downloading {files_to_download} files ({total_size:.1f}MB total)")
        
        # Download with progress bar
        downloaded_count = 0
        with self.logger.progress_bar(files_to_download, f"Downloading batch files") as progress:
            task = progress.add_task("downloading", total=files_to_download)
            
            for key, size_mb in valid_files[:files_to_download]:
                dest_path = self.download_to / key
                
                # Create parent directories
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    self.client.download_file(self.bucket, key, str(dest_path))
                    self.count += 1
                    downloaded_count += 1
                    self.progress.append_dowloaded(key)
                    
                    step_context.log(f"✓ {key} ({size_mb:.1f}MB)")
                    
                except Exception as e:
                    step_context.log(f"✗ Failed to download {key}: {e}", LogLevel.ERROR)
                
                progress.advance(task)
                
                # Update step with current progress
                step_context.update({
                    "current_file": key,
                    "downloaded_this_batch": downloaded_count,
                    "total_downloaded": len(self.progress.downloaded_keys)
                })
        
        return downloaded_count

    def upload_metadata(self):
        """Upload metadata files to S3 with progress tracking"""
        
        with self.logger.step("metadata_upload", "Upload metadata to S3") as step:
            # Collect metadata files
            metadata_files = []
            total_size = 0
            
            step.log("Scanning for metadata files...")
            
            for root, dirs, files in os.walk(self.metadata_path):
                for file in files:
                    if file.endswith(('.json', '.csv', '.txt', '.scp')):
                        file_path = Path(root) / file
                        file_size = file_path.stat().st_size
                        metadata_files.append((root, file, file_size))
                        total_size += file_size
            
            if not metadata_files:
                step.log("No metadata files found to upload", LogLevel.WARN)
                step.update({"files_found": 0})
                return
            
            total_size_mb = total_size / 1024 / 1024
            step.update({
                "files_found": len(metadata_files),
                "total_size_mb": round(total_size_mb, 2)
            })
            
            step.log(f"Found {len(metadata_files)} metadata files ({total_size_mb:.1f}MB)")
            
            # Upload files with progress
            failed_uploads = []
            
            with self.logger.progress_bar(len(metadata_files), "Uploading metadata") as progress:
                task = progress.add_task("uploading", total=len(metadata_files))
                
                for i, (root, file, file_size) in enumerate(metadata_files):
                    local_path = os.path.join(root, file)
                    s3_key = os.path.join(self.metadata_prefix, file).replace("\\", "/")
                    
                    try:
                        file_size_kb = file_size / 1024
                        self.client.upload_file(local_path, self.bucket, s3_key)
                        step.log(f"✓ {file} ({file_size_kb:.1f}KB)")
                        
                    except Exception as e:
                        failed_uploads.append(file)
                        step.log(f"✗ Failed to upload {file}: {e}", LogLevel.ERROR)
                    
                    progress.advance(task)
                    step.update({
                        "current_file": file,
                        "uploaded_count": i + 1 - len(failed_uploads)
                    })
            
            # Final results
            success_count = len(metadata_files) - len(failed_uploads)
            step.update({
                "uploaded_successfully": success_count,
                "failed_uploads": len(failed_uploads)
            })
            
            if failed_uploads:
                step.log(f"Upload completed with {len(failed_uploads)} failures", LogLevel.WARN)
            else:
                step.log(f"All {success_count} metadata files uploaded successfully")

    def upload(self):
        """Upload processed files to S3 with detailed progress"""
        
        wave_paths = list(self.upload_from.glob("*.wav"))
        batch_num = self.progress.batch_count + 1
        
        with self.logger.step(f"batch_upload_{batch_num}", f"Upload batch #{batch_num} results") as step:
            if not wave_paths:
                step.log("No .wav files found to upload", LogLevel.WARN)
                step.update({"files_to_upload": 0})
                return
            
            # Calculate total size
            total_size = sum(p.stat().st_size for p in wave_paths)
            total_size_mb = total_size / 1024 / 1024
            
            step.log(f"Found {len(wave_paths)} processed files to upload ({total_size_mb:.1f}MB)")
            step.update({
                "files_to_upload": len(wave_paths),
                "total_size_mb": round(total_size_mb, 2)
            })
            
            # Upload files with progress
            failed_uploads = []
            uploaded_size = 0
            
            with self.logger.progress_bar(len(wave_paths), "Uploading processed files") as progress:
                task = progress.add_task("uploading", total=len(wave_paths))
                
                for i, audio_path in enumerate(wave_paths):
                    key = f"{self.processed_prefix}/{audio_path.name}"
                    file_size = audio_path.stat().st_size
                    file_size_mb = file_size / 1024 / 1024
                    
                    try:
                        self.client.upload_file(str(audio_path), self.bucket, key)
                        self.progress.append_uploaded(key)
                        uploaded_size += file_size
                        
                        step.log(f"✓ {audio_path.name} ({file_size_mb:.1f}MB)")
                        
                    except Exception as e:
                        failed_uploads.append(audio_path.name)
                        step.log(f"✗ Failed to upload {audio_path.name}: {e}", LogLevel.ERROR)
                    
                    progress.advance(task)
                    
                    # Update step with current progress
                    step.update({
                        "current_file": audio_path.name,
                        "uploaded_count": i + 1 - len(failed_uploads),
                        "uploaded_size_mb": round(uploaded_size / 1024 / 1024, 2)
                    })
            
            # Final results
            success_count = len(wave_paths) - len(failed_uploads)
            step.update({
                "uploaded_successfully": success_count,
                "failed_uploads": len(failed_uploads),
                "final_uploaded_size_mb": round(uploaded_size / 1024 / 1024, 2)
            })
            
            if failed_uploads:
                step.log(f"Upload completed with {len(failed_uploads)} failures", LogLevel.WARN)
                step.log(f"Failed files: {', '.join(failed_uploads)}")
            else:
                step.log(f"All {success_count} files uploaded successfully")

        # Update progress tracking
        with self.logger.step("progress_update", "Update progress tracking") as step:
            self.progress.increment_progress()
            self.progress.save()
            
            step.update({
                "batch_number": self.progress.batch_count,
                "total_downloaded": len(self.progress.downloaded_keys),
                "total_uploaded": len(self.progress.uploaded_keys),
                "total_chunks": self.progress.total_chunks
            })
            
            step.log("Progress tracking updated")
            
        # Show progress report
        self.progress.report()

    def clear_local_dirs(self):
        """Reset the local download and upload directories for next batch."""
        for path in (self.download_to, self.upload_from):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    def empty_batch(self):
        """Mark batch as complete"""
        self.state.token = None
        self.state.completed = True

    def to_params(self) -> dict:
        """Convert the batcher state to parameters for S3 operations."""
        params = {
            "Bucket": self.bucket,
            "MaxKeys": 1000,
        }

        if self.state.has_token():
            params["ContinuationToken"] = self.state.token

        return params

    def counter(self, prefix: str | None = None) -> int:
        """Count the number of objects in the S3 bucket with the given prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        total = 0

        kwargs = {"Bucket": self.bucket}
        if prefix:
            kwargs["Prefix"] = prefix

        try:
            for page in paginator.paginate(**kwargs, PaginationConfig={"PageSize": 1000}):
                contents = page.get("Contents", [])
                # Apply same filters as download to get accurate count
                for obj in contents:
                    key = obj["Key"]
                    if self.include_ext and not key.lower().endswith(self.include_ext):
                        continue
                    if any(key.startswith(p) for p in self.ignore_prefixes):
                        continue
                    total += 1
        except Exception as e:
            self.logger.error(f"Error counting S3 objects: {e}")
            return 0

        return total

    def format_token(self, token: str | None) -> str:
        """Format continuation token for display"""
        if not token:
            return "None"
        return f"{token[:20]}..." if len(token) > 20 else token
