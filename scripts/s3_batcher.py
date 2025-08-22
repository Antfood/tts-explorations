import boto3
from pathlib import Path
import json
from typing import List
from .progress import ProgressInfo
import shutil


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
        batch_size: int = 25,
        include_ext: tuple[str, ...] = (".wav",),
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
        self.count = 0

        self.total = self.counter()
        self.progress = ProgressInfo(metadata_path, self.total)
        print(
            f":: Found {self.total} total files in s3://{self.bucket}/{self.processed_prefix}"
        )

    def has_next(self) -> bool:
        """Check if there are more files to process."""
        return not self.state.completed

    def next_batch(self):
        """Get the next batch of files to process from S3."""
        self.count = 0
        self.clear_local_dirs()
        # pull progress from disk
        self.progress = ProgressInfo(self.metadata_path, self.total)
        print(f":: Fetching next batch  {self.progress.batch_count} from S3.")
        print(f":: Current token: {self.state.token}")

        while self.count < self.batch_size:

            params = self.to_params()
            resp = self.client.list_objects_v2(**params)

            self.state.token_from_response(resp)
            batch = resp.get("Contents", [])

            if not batch or len(batch) == 0:
                print(":: No more files to process.")
                self.empty_batch()
                break

            completed = self.save_batch(batch)

            if completed:
                break

            if len(batch) <= self.batch_size:
                print( f":: Batch {self.progress.batch_count} completed with {len(batch)} files.")
                break

        print(f":: Batch {self.progress.batch_count} has downloaded {self.count} files.")
        self.state.save()

    def upload(self):
        wave_paths = list(self.upload_from.glob("*.wav"))
        print(
            f":: Batch {self.progress.batch_count} done. Uploading processed files to S3."
        )

        if not wave_paths:
            print(f":: No .wav files found in {self.upload_from}, skipping upload.")
            return

        for i, audio_path in enumerate(wave_paths):
            key = f"{self.processed_prefix}/{audio_path.name}"

            try:
                self.client.upload_file(str(audio_path), self.bucket, key)
                self.progress.append_uploaded(key)
                print(f":: Uploaded {i + 1}/{len(wave_paths)}")

            except Exception as e:
                print(f":: Error uploading {audio_path}: {e}")

        print(f":: Uploaded {len(wave_paths)} files to S3. Incrementing progress.")
        self.progress.increment_progress()
        self.progress.save()
        self.progress.report()

    def clear_local_dirs(self):
        """Reset the local download and upload directories for next batch."""

        for path in (self.download_to, self.upload_from):
            if path.exists():
                shutil.rmtree(path)

            path.mkdir(parents=True, exist_ok=True)

    def empty_batch(self):
        self.state.token = None
        self.state.completed = True

    def save_batch(self, batch: List[dict]) -> bool:
        """download and save the current batch files to local disk.
        Returns True if the batch is complete, False otherwise.
        """

        for obj in batch:
            key = obj["Key"]

            if self.include_ext and not key.lower().endswith(self.include_ext):
                print(
                    f":: Skipping {key}, does not match include_ext {self.include_ext}"
                )
                continue

            dest_path = self.download_to / key

            try:
                print(f":: [{self.count + 1}/{self.batch_size}] Downloading {key} to {dest_path}")
                self.client.download_file(self.bucket, key, str(dest_path))
                self.count += 1
                self.progress.append_dowloaded(key)

            except Exception as e:
                print(f":: Error downloading {key}: {e}")
                continue

            if self.count >= self.batch_size:
                return True

        return False

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

        for page in paginator.paginate(**kwargs, PaginationConfig={"PageSize": 1000}):
            total += len(page.get("Contents", []))

        return total
