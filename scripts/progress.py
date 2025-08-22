from pathlib import Path
import json
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn


def _human(n: int) -> str:
    r = n

    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n}{unit}"

        r = round(n / 1000, 1)

    return f"{r}P"


class ProgressInfo:
    def __init__(
        self, metadata_path: Path, total_expected: int, name: str = "progress.json"
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

    def append_dowloaded(self, key: str):
        """Append a downloaded key to the list"""
        self.downloaded_keys.append(key)

    def append_uploaded(self, key: str):
        """Append an uploaded key to the list"""
        self.uploaded_keys.append(key)
        self.total_chunks += 1

    def increment_batch(self):
        self.batch_count += 1
        self.before_process_count = len(self.downloaded_keys)
        self.after_process_count = len(self.uploaded_keys)

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
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        with open(self.file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self):
        if not self.file.exists():
            return
        with open(self.file, "r") as f:
            data = json.load(f)
        self.downloaded_keys = data.get("downloaded_keys", [])
        self.uploaded_keys = data.get("uploaded_keys", [])
        self.total_expected = data.get("total_expected", self.total_expected)
        self.total_chunks = data.get("total_chunks", 0)
        self.batch_count = data.get("batch_count", 0)
        self.before_process_count = data.get("before_process_count", 0)
        self.after_process_count = data.get("after_process_count", 0)

    def report(self):
        """
        Pretty Report
        """
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
        table.add_row("Files downloaded", f"{d} ({_human(d)})")
        table.add_row("Files uploaded", f"{u} ({_human(u)})")
        table.add_row("Total chunks", f"{chunks} ({_human(chunks)})")
        table.add_row("Files before processing", str(self.before_process_count))
        table.add_row("Files after processing", str(self.after_process_count))

        console.print("\n[bold]Progress Report[/bold]")
        console.print(table)

        if self.total_expected:
            console.print()  # spacer
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task("Downloads", total=self.total_expected)
                progress.update(task_id, completed=d)

            console.print()  
