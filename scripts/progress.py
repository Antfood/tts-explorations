from pathlib import Path
import json
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn

from .logger import PrettyLogger, LogLevel


def _human(n: int) -> str:
    r = n

    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n}{unit}"

        r = round(n / 1000, 1)

    return f"{r}P"


class ProgressInfo:
    def __init__(
        self, 
        metadata_path: Path, 
        total_expected: int, 
        name: str = "progress.json",
        logger: Optional[PrettyLogger] = None
    ):
        self.file = metadata_path / name
        self.downloaded_keys: List[str] = []
        self.uploaded_keys: List[str] = []
        self.total_expected: int = total_expected
        self.metadata_path: Path = metadata_path
        self.total_chunks: int = 0
        self.batch_count: int = 0
        self.before_process_count: int = 0
        self.after_process_count: int = 0
        self.logger = logger or PrettyLogger()

    def append_dowloaded(self, key: str):
        """Append a downloaded key to the list"""
        self.downloaded_keys.append(key)
        self.logger.debug(f"Marked as downloaded: {key}")

    def append_uploaded(self, key: str):
        """Append an uploaded key to the list"""
        self.uploaded_keys.append(key)
        self.total_chunks += 1
        self.logger.debug(f"Marked as uploaded: {key} (chunk #{self.total_chunks})")

    def increment_progress(self):
        """Increment batch progress and log the update"""
        old_batch = self.batch_count
        self.batch_count += 1
        self.before_process_count = len(self.downloaded_keys)
        self.after_process_count = len(self.uploaded_keys)
        
        self.logger.debug(f"Progress incremented: batch {old_batch} → {self.batch_count}")
        self.logger.debug(f"Downloaded: {self.before_process_count}, Uploaded: {self.after_process_count}")

    def to_dict(self):
        return {
            "downloaded_keys": self.downloaded_keys,
            "uploaded_keys": self.uploaded_keys,
            "total_expected": self.total_expected,
            "total_chunks": self.total_chunks,
            "batch_count": self.batch_count,
            "before_process_count": self.before_process_count,
            "after_process_count": self.after_process_count,
        }

    def save(self):
        """Save progress to disk with logging"""
        with self.logger.step("progress_save", "Save progress to disk") as step:
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            
            with open(self.file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            
            step.update({
                "progress_file": str(self.file),
                "batch_count": self.batch_count,
                "downloaded_count": len(self.downloaded_keys),
                "uploaded_count": len(self.uploaded_keys),
                "total_chunks": self.total_chunks
            })
            
            step.log(f"Progress saved: batch #{self.batch_count}")

    def load(self):
        """Load progress from disk with logging"""
        if not self.file.exists():
            self.logger.debug(f"No existing progress file found at {self.file}")
            return

        with self.logger.step("progress_load", "Load existing progress") as step:
            try:
                with open(self.file, "r") as f:
                    data = json.load(f)

                self.downloaded_keys = data.get("downloaded_keys", [])
                self.uploaded_keys = data.get("uploaded_keys", [])
                self.total_expected = data.get("total_expected", self.total_expected)
                self.total_chunks = data.get("total_chunks", 0)
                self.batch_count = data.get("batch_count", 0)
                self.before_process_count = data.get("before_process_count", 0)
                self.after_process_count = data.get("after_process_count", 0)
                
                step.update({
                    "loaded_batch_count": self.batch_count,
                    "loaded_downloads": len(self.downloaded_keys),
                    "loaded_uploads": len(self.uploaded_keys),
                    "loaded_chunks": self.total_chunks
                })
                
                if self.batch_count > 0:
                    step.log(f"Resumed from batch #{self.batch_count}")
                else:
                    step.log("Starting fresh (no previous progress)")
                    
            except Exception as e:
                step.log(f"Error loading progress file: {e}", LogLevel.ERROR)
                self.logger.warn("Starting with fresh progress tracking")

    def report(self):
        """
        Pretty Report with enhanced formatting
        """
        with self.logger.step("progress_report", "Generate progress report") as step:
            d, u, chunks = (
                len(self.downloaded_keys),
                len(self.uploaded_keys),
                self.total_chunks,
            )

            console = Console()
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="bold")

            table.add_row("Batches processed", str(self.batch_count))
            table.add_row("Files downloaded", f"{d:,} ({_human(d)})")
            table.add_row("Files uploaded", f"{u:,} ({_human(u)})")
            table.add_row("Total chunks", f"{chunks:,} ({_human(chunks)})")
            table.add_row("Files before processing", f"{self.before_process_count:,}")
            table.add_row("Files after processing", f"{self.after_process_count:,}")

            console.print("\n[bold]Progress Report[/bold]")
            console.print(table)

            # Calculate completion percentage if we have total expected
            completion_rate = 0
            if self.total_expected > 0:
                completion_rate = (d / self.total_expected) * 100
                
                console.print()  # spacer
                with Progress(
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    transient=True,
                ) as progress:
                    task_id = progress.add_task("Overall Progress", total=self.total_expected)
                    progress.update(task_id, completed=d)

                console.print()
            
            step.update({
                "downloads": d,
                "uploads": u, 
                "chunks": chunks,
                "completion_percentage": round(completion_rate, 1) if self.total_expected > 0 else 0
            })
            
            step.log(f"Report generated: {chunks:,} chunks from {self.batch_count} batches")
