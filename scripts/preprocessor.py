from pathlib import Path
import whisperx
import torch
import librosa
import soundfile as sf
import os
import re
from typing import List, Optional
from dataclasses import dataclass
from . import constants as const
from .logger import PrettyLogger, LogLevel

from .normalizer.text_normalizer import TextNormalizer

@dataclass
class ProcessedChunk:
    original_audio_path: Path
    text_path: Path
    audio_path: Path
    text: str
    chunk_index: int

    def to_list(self):
        return [
            self.original_audio_path,
            self.text_path,
            self.audio_path,
            self.text,
            self.chunk_index,
        ]

    @staticmethod
    def headers():
        return [
            "original_audio_path",
            "text_path",
            "audio_path",
            "text",
            "chunk_index",
        ]


os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"


class Preprocessor:
    def __init__(
        self,
        in_path: Path = const.DEFAULT_IN_PATH,
        out_path: Path = const.DEFAULT_OUT_PATH,
        metadata_path: Path = const.DEFAULT_METADATA_PATH,
        model_size: str = "small",
        compute_type: str = "float32",
        language: str = const.DEFAULT_LANGUAGE,
        batch_size: int = const.WHISPER_BATCH,
        target_sr: int = const.DEFAULT_TARGET_SR,
        logger: Optional[PrettyLogger] = None,
    ):
        self.in_path = in_path
        self.out_path = out_path
        self.metadata_path = metadata_path
        self.language = language
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_compute = compute_type
        self.target_sr = target_sr
        self.logger = logger or DockerLikeLogger()
        
        # Initialize models with enhanced logging
        with self.logger.step("whisper_init", "Initialize Whisper model") as step:
            step.log(f"Loading Whisper model: {model_size}")
            step.update({
                "model_size": model_size, 
                "device": self.device, 
                "compute_type": compute_type,
                "language": language
            })
            
            try:
                self.whisper_model = whisperx.load_model(
                    model_size,
                    device=self.device,
                    language=self.language,
                    compute_type=self.whisper_compute,
                )
                step.log("Whisper model loaded successfully")
                
            except Exception as e:
                step.log(f"Failed to load Whisper model: {e}", LogLevel.ERROR)
                raise

        with self.logger.step("align_init", "Initialize alignment model") as step:
            step.log(f"Loading alignment model for language: {language}")
            step.update({"language": language, "device": self.device})
            
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                )
                step.log("Alignment model loaded successfully")
                
            except Exception as e:
                step.log(f"Failed to load alignment model: {e}", LogLevel.ERROR)
                raise

        self.logger.info(f"Preprocessor initialized - Device: {self.device}")
        self.normalizer = TextNormalizer()

    def set_language(self, language: str):
        """Change the language for alignment model"""
        if self.language == language:
            self.logger.debug(f"Already using language {language}, skipping reload")
            return

        with self.logger.step("language_change", f"Change language to {language}") as step:
            step.log(f"Switching alignment model: {self.language} → {language}")
            step.update({
                "old_language": self.language, 
                "new_language": language,
                "device": self.device
            })
            
            try:
                self.language = language
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device,
                )
                step.log(f"Language change successful")
                
            except Exception as e:
                step.log(f"Failed to change language: {e}", LogLevel.ERROR)
                raise

    def transcribe(self, audio_path: Path):
        """Transcribe audio file"""
        self.logger.debug(f"Starting transcription: {audio_path.name}")
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path.absolute())
            audio_duration = len(audio) / 16000  # WhisperX uses 16kHz
            
            self.logger.debug(f"Audio loaded: {audio_duration:.1f}s, batch_size={self.batch_size}")
            
            # Transcribe
            transcription = self.whisper_model.transcribe(audio, batch_size=self.batch_size)
            
            # Log results
            segments = transcription.get("segments", [])
            detected_lang = transcription.get("language", "unknown")
            
            self.logger.debug(f"Transcription complete: {len(segments)} segments, language={detected_lang}")
            
            return audio, transcription
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path.name}: {e}")
            raise

    def align(self, audio, transcription) -> dict:
        """Perform forced alignment"""
        transcription_lang = transcription.get("language", "unknown")
        
        # Check for language mismatch
        if self.language != transcription_lang:
            self.logger.warn(f"Language mismatch - Align: {self.language}, Transcription: {transcription_lang}")
            self.logger.info("Reloading alignment model with detected language")
            self.set_language(transcription_lang)

        self.logger.debug("Starting forced alignment")
        
        try:
            segments = transcription.get("segments", [])
            if not segments:
                self.logger.warn("No segments found for alignment")
                return {"segments": []}
            
            aligned_result = whisperx.align(
                segments,
                self.align_model,
                self.align_metadata,
                audio,
                device=self.device,
                return_char_alignments=False,
            )
            
            aligned_segments = aligned_result.get("segments", [])
            self.logger.debug(f"Alignment complete: {len(aligned_segments)} segments aligned")
            
            return aligned_result
            
        except Exception as e:
            self.logger.error(f"Alignment failed: {e}")
            raise

    def split_audio(
        self,
        audio_path: Path,
        aligned_result: dict,
        target_sr: int = 44100,
    ) -> List[ProcessedChunk]:
        """Split audio into chunks based on alignment"""
        
        file_step_id = f"split_{audio_path.stem}"
        with self.logger.step(file_step_id, f"Split {audio_path.name} into chunks") as step:
            
            # Load audio
            step.log(f"Loading audio at {target_sr}Hz...")
            audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            
            segments = aligned_result.get("segments", [])
            if not segments:
                step.log("No segments found for splitting", LogLevel.WARN)
                return []
            
            step.update({
                "input_file": audio_path.name,
                "target_sr": target_sr,
                "segments_to_split": len(segments),
                "audio_duration": len(audio_data) / sr
            })
            
            step.log(f"Splitting into {len(segments)} chunks...")
            
            chunks = []
            failed_chunks = 0
            
            # Process each segment
            with self.logger.progress_bar(len(segments), f"Processing {audio_path.name}") as progress:
                task = progress.add_task("splitting", total=len(segments))
                
                for i, segment in enumerate(segments):
                    try:
                        start = segment.get("start", 0)
                        end = segment.get("end", start + 1)
                        text = segment.get("text", "").strip()
                        
                        if not text:
                            self.logger.debug(f"Skipping empty segment {i}")
                            progress.advance(task)
                            continue
                        
                        # Normalize text
                        normalized_text = self.normalizer.normalize_text(text, self.language)
                        
                        # Extract audio chunk
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        chunk = audio_data[start_sample:end_sample]
                        
                        if len(chunk) == 0:
                            self.logger.debug(f"Skipping zero-length chunk {i}")
                            progress.advance(task)
                            continue
                        
                        # Generate output paths
                        clean_name = re.sub(r"[^a-z0-9]", "_", audio_path.name.lower())
                        chunk_id = f"{clean_name}_chunk-{i:04d}"
                        
                        audio_out_path = self.out_path / f"{chunk_id}.wav"
                        text_out_path = self.metadata_path / "text"
                        wav_scp_path = self.metadata_path / "wav.scp"
                        
                        # Save audio chunk
                        sf.write(data=chunk, samplerate=sr, file=audio_out_path)
                        
                        # Append to metadata files
                        with open(text_out_path, "a", encoding="utf-8") as f:
                            f.write(f"{chunk_id} {normalized_text}\n")
                        
                        with open(wav_scp_path, "a", encoding="utf-8") as f:
                            rel_path = audio_out_path.relative_to(self.out_path)
                            f.write(f"{chunk_id} {rel_path}\n")
                        
                        # Create chunk object
                        processed = ProcessedChunk(
                            original_audio_path=audio_path,
                            chunk_index=i,
                            text=normalized_text,
                            audio_path=audio_out_path,
                            text_path=text_out_path,
                        )
                        
                        chunks.append(processed)
                        
                    except Exception as e:
                        failed_chunks += 1
                        self.logger.debug(f"Failed to process chunk {i}: {e}")
                    
                    progress.advance(task)
                    
                    # Update step with progress
                    step.update({
                        "chunks_created": len(chunks),
                        "current_chunk": i + 1,
                        "failed_chunks": failed_chunks
                    })
            
            step.update({
                "total_chunks_created": len(chunks),
                "failed_chunks": failed_chunks,
                "success_rate": round((len(chunks) / len(segments)) * 100, 1) if segments else 0
            })
            
            if failed_chunks > 0:
                step.log(f"Completed with {failed_chunks} failed chunks", LogLevel.WARN)
            else:
                step.log(f"Successfully created {len(chunks)} chunks")

        return chunks

    def preprocess(self):
        """Main preprocessing pipeline"""
        
        with self.logger.step("file_discovery", "Discover audio files") as step:
            wave_paths = list(self.in_path.glob("*.wav"))
            
            step.update({
                "search_path": str(self.in_path),
                "files_found": len(wave_paths)
            })
            
            if not wave_paths:
                step.log("No .wav files found", LogLevel.WARN)
                step.log(f"Searched in: {self.in_path}")
                return
            
            step.log(f"Found {len(wave_paths)} audio files to process")
            
            # Log file details
            total_size = sum(p.stat().st_size for p in wave_paths) / 1024 / 1024  # MB
            step.update({"total_size_mb": round(total_size, 2)})
        
        total_chunks_created = 0
        failed_files = []
        
        # Process each file
        with self.logger.progress_bar(len(wave_paths), "Processing audio files") as progress:
            main_task = progress.add_task("files", total=len(wave_paths))
            
            for i, audio_path in enumerate(wave_paths):
                file_step_id = f"process_file_{i}"
                
                try:
                    with self.logger.step(file_step_id, f"Process {audio_path.name}") as step:
                        file_size_mb = audio_path.stat().st_size / 1024 / 1024
                        step.update({
                            "file_index": i + 1,
                            "total_files": len(wave_paths),
                            "file_size_mb": round(file_size_mb, 2)
                        })
                        
                        # Transcription
                        step.log("Starting transcription...")
                        audio, transcription = self.transcribe(audio_path)
                        
                        detected_lang = transcription.get("language", "unknown")
                        segments = transcription.get("segments", [])
                        
                        step.update({
                            "detected_language": detected_lang,
                            "transcription_segments": len(segments)
                        })
                        step.log(f"Transcription: {len(segments)} segments, language: {detected_lang}")
                        
                        # Alignment
                        step.log("Starting alignment...")
                        aligned_results = self.align(audio, transcription)
                        
                        # Ensure output directory exists
                        self.out_path.mkdir(parents=True, exist_ok=True)
                        
                        # Split audio
                        step.log("Splitting audio into chunks...")
                        chunks = self.split_audio(
                            audio_path,
                            aligned_results,
                            target_sr=self.target_sr,
                        )
                        
                        chunk_count = len(chunks)
                        total_chunks_created += chunk_count
                        
                        step.update({
                            "chunks_created": chunk_count,
                            "running_total_chunks": total_chunks_created
                        })
                        
                        step.log(f"File complete: {chunk_count} chunks created")
                        
                        yield chunks
                        
                except Exception as e:
                    failed_files.append(audio_path.name)
                    self.logger.error(f"Failed to process {audio_path.name}: {e}")
                    # Continue with next file instead of crashing
                    yield []  # Return empty chunks for failed file
                
                progress.advance(main_task)
        
        # Final summary
        with self.logger.step("preprocessing_summary", "Preprocessing complete") as step:
            success_count = len(wave_paths) - len(failed_files)
            
            step.update({
                "total_files": len(wave_paths),
                "successful_files": success_count,
                "failed_files": len(failed_files),
                "total_chunks_created": total_chunks_created,
                "success_rate": round((success_count / len(wave_paths)) * 100, 1) if wave_paths else 0
            })
            
            if failed_files:
                step.log(f"Processing complete with {len(failed_files)} failures", LogLevel.WARN)
                step.log(f"Failed files: {', '.join(failed_files)}")
            else:
                step.log("All files processed successfully")
            
            step.log(f"Total output: {total_chunks_created:,} audio chunks")

    def init_files(self):
        """Initialize output files"""
        with self.logger.step("file_init", "Initialize output files") as step:
            text_path = self.metadata_path / "text"
            wav_scp_path = self.metadata_path / "wav.scp"
            
            # Ensure metadata directory exists
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            
            # Create empty files
            text_path.write_text("", encoding="utf-8")
            wav_scp_path.write_text("", encoding="utf-8")
            
            step.update({
                "text_file": str(text_path),
                "wav_scp_file": str(wav_scp_path),
                "metadata_dir": str(self.metadata_path)
            })
            
            step.log("Output files initialized")
