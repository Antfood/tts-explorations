import time
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class StepInfo:
    name: str
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_running(self) -> bool:
        return self.status == "running"
    
    @property
    def is_complete(self) -> bool:
        return self.status in ["complete", "success", "error"]


class PrettyLogger:
    def __init__(self):
        self.console = Console()
        self.steps: Dict[str, StepInfo] = {}
        self.current_step: Optional[str] = None
        self.start_time = time.time()
        self.live_display: Optional[Live] = None
        
        # Style configuration
        self.styles = {
            LogLevel.DEBUG: "dim white",
            LogLevel.INFO: "bright_blue",
            LogLevel.WARN: "bright_yellow",
            LogLevel.ERROR: "bright_red",
            LogLevel.SUCCESS: "bright_green"
        }
        
        self.status_styles = {
            "pending": "dim white",
            "running": "bright_yellow",
            "complete": "bright_green",
            "success": "bright_green", 
            "error": "bright_red",
            "skipped": "dim yellow"
        }
    
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO, prefix: str = ""):
        timestamp = self._get_timestamp()
        style = self.styles.get(level, "white")
        
        # Format like Docker: timestamp [level] message
        level_text = f"[{level.value.upper():>5}]"
        prefix_text = f" {prefix}" if prefix else ""
        
        formatted_message = Text()
        formatted_message.append(f"{timestamp} ", style="dim white")
        formatted_message.append(level_text, style=style)
        formatted_message.append(prefix_text, style="dim white")
        formatted_message.append(f" {message}")
        
        self.console.print(formatted_message)
    
    def step(self, step_id: str, name: str, details: Optional[Dict[str, Any]] = None):
        """Start a new step"""
        self.steps[step_id] = StepInfo(
            name=name,
            status="running",
            start_time=time.time(),
            details=details or {}
        )
        self.current_step = step_id
        
        # Log step start
        self.log(f"Step {len(self.steps)}: {name}", LogLevel.INFO, "→")
        
        # Update live display if active
        self.update_live_display()
        
        return StepContext(self, step_id)
    
    def update_step(self, step_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Update step status and details"""
        if step_id in self.steps:
            step = self.steps[step_id]
            step.status = status
            if status in ["complete", "success", "error"]:
                step.end_time = time.time()
            if details:
                step.details.update(details)
            
            # Update live display if active
            self.update_live_display()
    
    def complete_step(self, step_id: str, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Mark step as complete"""
        if step_id in self.steps:
            step = self.steps[step_id]
            step.status = "success"
            step.end_time = time.time()
            if details:
                step.details.update(details)
            
            duration_text = f" ({self._format_duration(step.duration)})" if step.duration else ""
            final_message = message or f"Completed: {step.name}"
            self.log(f"{final_message}{duration_text}", LogLevel.SUCCESS, "✓")
            
            # Update live display if active
            self.update_live_display()
    
    def error_step(self, step_id: str, error: str):
        """Mark step as error"""
        if step_id in self.steps:
            step = self.steps[step_id]
            step.status = "error"
            step.end_time = time.time()
            step.details["error"] = error
            
            duration_text = f" ({self._format_duration(step.duration)})" if step.duration else ""
            self.log(f"Failed: {step.name} - {error}{duration_text}", LogLevel.ERROR, "✗")
            
            # Update live display if active
            self.update_live_display()
    
    def progress_bar(self, total: int, description: str = "Processing"):
        """Create a progress bar context manager"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        )
    
    @contextmanager
    def live_status_panel(self, title: str = "Processing Status"):
        """Create a live-updating status panel for long-running operations"""
        def generate_status_table():
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Step", style="cyan", no_wrap=True)
            table.add_column("Status", justify="center", min_width=10)
            table.add_column("Duration", justify="right")
            table.add_column("Details", max_width=40)
            
            for step_id, step in self.steps.items():
                status_text = Text(step.status.title())
                status_text.stylize(self.status_styles.get(step.status, "white"))
                
                if step.is_running and step.start_time:
                    duration = self._format_duration(time.time() - step.start_time)
                elif step.duration:
                    duration = self._format_duration(step.duration)
                else:
                    duration = "-"
                
                # Format details
                details_str = ""
                if step.details:
                    key_details = []
                    for key, value in step.details.items():
                        if key != "error":
                            if isinstance(value, (int, float)) and key.endswith(("_count", "_size", "_total")):
                                key_details.append(f"{key}: {value:,}")
                            else:
                                key_details.append(f"{key}: {value}")
                    details_str = ", ".join(key_details[:2])  # Limit to 2 details for space
                    if len(step.details) > 2:
                        details_str += "..."
                
                table.add_row(step.name, status_text, duration, details_str)
            
            return Panel(table, title=title, border_style="blue")
        
        self.live_display = Live(generate_status_table(), console=self.console, refresh_per_second=2)
        self.live_display.start()
        
        try:
            yield self.live_display
        finally:
            self.live_display.stop()
            self.live_display = None
    
    def update_live_display(self):
        """Update the live display if it's active"""
        if self.live_display:
            def generate_status_table():
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Step", style="cyan", no_wrap=True)
                table.add_column("Status", justify="center", min_width=10)
                table.add_column("Duration", justify="right")
                table.add_column("Details", max_width=40)
                
                for step_id, step in self.steps.items():
                    status_text = Text(step.status.title())
                    status_text.stylize(self.status_styles.get(step.status, "white"))
                    
                    if step.is_running and step.start_time:
                        duration = self._format_duration(time.time() - step.start_time)
                    elif step.duration:
                        duration = self._format_duration(step.duration)
                    else:
                        duration = "-"
                    
                    # Format details
                    details_str = ""
                    if step.details:
                        key_details = []
                        for key, value in step.details.items():
                            if key != "error":
                                if isinstance(value, (int, float)) and key.endswith(("_count", "_size", "_total")):
                                    key_details.append(f"{key}: {value:,}")
                                else:
                                    key_details.append(f"{key}: {value}")
                        details_str = ", ".join(key_details[:2])
                        if len(step.details) > 2:
                            details_str += "..."
                    
                    table.add_row(step.name, status_text, duration, details_str)
                
                return Panel(table, title="Processing Status", border_style="blue")
            
            self.live_display.update(generate_status_table())
    
    def print_summary(self):
        """Print execution summary"""
        total_duration = time.time() - self.start_time
        
        # Create summary table
        table = Table(title="Execution Summary", show_header=True)
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Details")
        
        for step_id, step in self.steps.items():
            status_text = Text(step.status.title())
            status_text.stylize(self.status_styles.get(step.status, "white"))
            
            duration = self._format_duration(step.duration) if step.duration else "-"
            
            # Format details
            details_str = ""
            if step.details:
                key_details = []
                for key, value in step.details.items():
                    if key != "error":  # Error shown in status
                        if isinstance(value, (int, float)) and key.endswith(("_count", "_size", "_total")):
                            key_details.append(f"{key}: {value:,}")
                        else:
                            key_details.append(f"{key}: {value}")
                details_str = ", ".join(key_details)
            
            table.add_row(step.name, status_text, duration, details_str)
        
        # Add total row
        table.add_row(
            "[bold]Total[/bold]", 
            "", 
            f"[bold]{self._format_duration(total_duration)}[/bold]",
            ""
        )
        
        # Wrap the table in a panel for better presentation
        summary_panel = Panel(
            table, 
            title="🎯 Pipeline Summary", 
            border_style="green" if all(step.status == "success" for step in self.steps.values()) else "yellow"
        )
        
        self.console.print()
        self.console.print(summary_panel)
    
    @contextmanager 
    def panel_section(self, title: str, style: str = "blue"):
        """Create a visually distinct section with a panel"""
        self.console.print()
        panel = Panel(f"[bold]{title}[/bold]", style=style)
        self.console.print(panel)
        
        yield
        
        # Optional: Print completion panel
        completion_panel = Panel(f"[bold green]✓ {title} Complete[/bold green]", style="green")
        self.console.print(completion_panel)
    
    def info(self, message: str):
        self.log(message, LogLevel.INFO)
    
    def warn(self, message: str):
        self.log(message, LogLevel.WARN)
    
    def error(self, message: str):
        self.log(message, LogLevel.ERROR)
    
    def success(self, message: str):
        self.log(message, LogLevel.SUCCESS)
    
    def debug(self, message: str):
        self.log(message, LogLevel.DEBUG)


class StepContext:
    """Context manager for steps"""
    def __init__(self, logger: PrettyLogger, step_id: str):
        self.logger = logger
        self.step_id = step_id
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.complete_step(self.step_id)
        else:
            self.logger.error_step(self.step_id, str(exc_val))
        return False  # Don't suppress exceptions
    
    def update(self, details: Dict[str, Any]):
        """Update step details"""
        self.logger.update_step(self.step_id, "running", details)
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        """Log within step context"""
        step_name = self.logger.steps[self.step_id].name
        prefix = f"[{step_name}]"
        self.logger.log(message, level, prefix)


# Usage example functions that show how to integrate with your existing code
def example_integration():
    """Example showing how to integrate with existing preprocessing code"""
    
    logger = PrettyLogger()
    logger.info("Starting TTS preprocessing pipeline")
    
    # Step 1: Setup
    with logger.step("setup", "Initialize models and configurations") as step:
        step.log("Loading Whisper model...")
        time.sleep(1)  # Simulate work
        step.update({"model_size": "small", "device": "cuda"})
        step.log("Loading alignment model...")
        time.sleep(0.5)
        step.update({"language": "pt"})
    
    # Step 2: S3 operations  
    with logger.step("s3_download", "Download batch from S3") as step:
        step.log("Fetching file list from S3...")
        time.sleep(0.3)
        
        # Simulate file processing with progress bar
        files = ["file1.wav", "file2.wav", "file3.wav", "file4.wav", "file5.wav"]
        step.update({"batch_size": len(files)})
        
        with logger.progress_bar(len(files), "Downloading files") as progress:
            task = progress.add_task("downloading", total=len(files))
            
            for i, file in enumerate(files):
                step.log(f"Downloaded {file}")
                time.sleep(0.2)  # Simulate download
                progress.advance(task)
        
        step.update({"downloaded_count": len(files), "total_mb": 150.5})
    
    # Step 3: Processing
    with logger.step("processing", "Process audio files") as step:
        step.log("Starting transcription...")
        
        with logger.progress_bar(len(files), "Processing audio") as progress:
            task = progress.add_task("transcribing", total=len(files))
            
            total_chunks = 0
            for i, file in enumerate(files):
                chunks = i + 2  # Simulate variable chunks per file
                total_chunks += chunks
                step.log(f"Processed {file} → {chunks} chunks")
                time.sleep(0.3)
                progress.advance(task)
        
        step.update({"files_processed": len(files), "total_chunks": total_chunks})
    
    # Step 4: Upload
    with logger.step("s3_upload", "Upload processed files") as step:
        step.log("Uploading processed audio chunks...")
        time.sleep(1)
        step.update({"uploaded_files": total_chunks, "upload_mb": 89.2})
        step.log("Uploading metadata...")
        time.sleep(0.2)
    
    logger.success("Pipeline completed successfully!")
    logger.print_summary()

def show_config_summary(args, logger: PrettyLogger):
    """Display a comprehensive summary of all configuration settings"""
    with logger.step("config_summary", "Configuration Summary") as step:
        
        # Create configuration table
        from rich.table import Table
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # Main configuration table
        config_table = Table(title="Pipeline Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Setting", style="cyan", no_wrap=True, min_width=20)
        config_table.add_column("Value", style="bright_white", min_width=30)
        config_table.add_column("Description", style="dim white")
        
        # File paths section
        config_table.add_row("📁 INPUT PATH", str(args.in_path), "Source directory for audio files")
        config_table.add_row("📁 OUTPUT PATH", str(args.out_path), "Destination for processed chunks")
        config_table.add_row("📁 METADATA PATH", str(args.metadata_path), "Location for metadata files")
        config_table.add_row("📄 CSV FILENAME", str(args.csv_filename), "Output CSV with chunk information")
        
        # Add separator
        config_table.add_row("", "", "")
        
        # Processing configuration
        config_table.add_row("🎤 WHISPER MODEL", args.whisper_size.upper(), f"Model size for transcription")
        config_table.add_row("🌍 LANGUAGE", args.lan.upper(), "Target language for processing")
        config_table.add_row("📊 WHISPER BATCH SIZE", str(args.whisper_batch_size), "Transcription batch size")
        config_table.add_row("🔊 TARGET SAMPLE RATE", f"{args.target_sr:,} Hz", "Output audio sample rate")
        
        # Add separator  
        config_table.add_row("", "", "")
        
        # S3 configuration
        config_table.add_row("☁️ S3 BUCKET", args.s3_bucket, "AWS S3 bucket name")
        config_table.add_row("📂 PROCESSED PREFIX", args.s3_processed_prefix, "S3 prefix for output files")
        config_table.add_row("📦 S3 BATCH SIZE", str(args.s3_batch_size), "Files per S3 download batch")
        
        # Add separator
        config_table.add_row("", "", "")
        
        # Runtime options
        mode_flags = []
        if args.only_meta:
            mode_flags.append("METADATA-ONLY")
        if args.dry_run:
            mode_flags.append("DRY-RUN")
        if args.verbose:
            mode_flags.append("VERBOSE")
        if args.live_dashboard:
            mode_flags.append("LIVE-DASHBOARD")
        
        mode_display = " | ".join(mode_flags) if mode_flags else "NORMAL"
        config_table.add_row("⚙️ RUNTIME MODE", mode_display, "Processing mode and flags")
        
        # Display the table in a panel
        config_panel = Panel(
            config_table,
            title="🔧 Processing Configuration",
            border_style="blue",
            padding=(1, 2)
        )
        
        console.print()
        console.print(config_panel)
        
        # Additional info section
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("", style="dim cyan", no_wrap=True)
        info_table.add_column("", style="dim white")
        
        # Calculate some derived values
        whisper_models = {
            "tiny": "~39MB, fastest, lowest quality",
            "base": "~74MB, fast, basic quality", 
            "small": "~244MB, balanced speed/quality",
            "medium": "~769MB, slower, better quality",
            "large": "~1550MB, slowest, best quality"
        }
        
        model_info = whisper_models.get(args.whisper_size, "Unknown model size")
        
        info_table.add_row("Model Details:", f"{model_info}")
        info_table.add_row("Expected Output:", f"44.1kHz mono WAV files + metadata")
        
        if not args.dry_run and not args.only_meta:
            info_table.add_row("Pipeline Flow:", "S3 Download → Transcribe → Align → Split → Upload")
        elif args.dry_run:
            info_table.add_row("Pipeline Flow:", "S3 Download → [SIMULATION ONLY]")
        elif args.only_meta:
            info_table.add_row("Pipeline Flow:", "Metadata Upload Only")
        
        # Memory/performance warnings
        warnings = []
        if args.whisper_batch_size > 16:
            warnings.append(f"Large batch size ({args.whisper_batch_size}) may cause GPU memory issues")
        if args.target_sr > 48000:
            warnings.append(f"High sample rate ({args.target_sr}Hz) will create large files")
        if args.s3_batch_size > 500:
            warnings.append(f"Large S3 batch size ({args.s3_batch_size}) may be slow to download")
        
        if warnings:
            info_table.add_row("⚠️ Warnings:", warnings[0])
            for warning in warnings[1:]:
                info_table.add_row("", warning)
        
        info_panel = Panel(
            info_table,
            title="ℹ️ Additional Information", 
            border_style="yellow",
            padding=(1, 2)
        )
        
        console.print(info_panel)
        console.print()
        
        # Update step with key configuration
        step.update({
            "whisper_model": args.whisper_size,
            "language": args.lan,
            "target_sample_rate": args.target_sr,
            "s3_bucket": args.s3_bucket,
            "whisper_batch_size": args.whisper_batch_size,
            "s3_batch_size": args.s3_batch_size,
            "runtime_mode": mode_display,
            "input_path": str(args.in_path),
            "output_path": str(args.out_path)
        })
        
        step.log("Configuration summary displayed")


if __name__ == "__main__":
    example_integration()
