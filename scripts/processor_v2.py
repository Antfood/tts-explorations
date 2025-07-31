from pathlib import Path
import whisperx
import torch
import librosa
import soundfile as sf
import re
import gc
import json
import time
from typing import List
from dataclasses import dataclass

@dataclass
class ProcessedChunk:
    original_audio_path: Path
    text_path: Path
    audio_path: Path
    text: str
    chunk_index: int
    start_time: float
    end_time: float

    def to_list(self):
        return [self.original_audio_path, self.text_path, self.audio_path, self.text, self.chunk_index, self.start_time, self.end_time]

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return allocated
    return 0

class OptimizedGPUProcessor:
    def __init__(self, force_cpu=False):
        # GPU Configuration - FIXED
        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        self.compute_type = "float32" if self.device == "cpu" else "float16"
        
        # Model Configuration
        self.model_size = "tiny" if self.device == "cpu" else "small"  # Use small on GPU
        self.batch_size = 1 if self.device == "cpu" else 8  # Larger batch on GPU
        
        # Models
        self.whisper_model = None
        self.align_models = {}  # Cache alignment models by language
        
        print(f"🚀 Initialized Processor:")
        print(f"   Device: {self.device}")
        print(f"   Compute Type: {self.compute_type}")
        print(f"   Model Size: {self.model_size}")
        print(f"   Batch Size: {self.batch_size}")
        
        # Load Whisper model once
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model with proper GPU utilization"""
        print(f"Loading Whisper {self.model_size} model on {self.device}...")
        try:
            self.whisper_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type
            )
            print("✅ Whisper model loaded")
            check_memory()
        except Exception as e:
            print(f"❌ GPU model failed, falling back to CPU: {e}")
            self.device = "cpu"
            self.compute_type = "float32"
            self.batch_size = 1
            self.whisper_model = whisperx.load_model("tiny", "cpu", compute_type="float32")
    
    def _get_alignment_model(self, language: str):
        """Get cached alignment model or load new one"""
        if language in self.align_models:
            return self.align_models[language]
        
        try:
            print(f"   Loading alignment model for {language}...")
            align_model, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            self.align_models[language] = (align_model, metadata)
            print(f"   ✅ Alignment model for {language} loaded")
            return align_model, metadata
        except Exception as e:
            print(f"   ⚠️ Alignment model for {language} failed: {e}")
            return None, None
    
    def process_single_file(self, audio_path: Path, out_dir: Path, target_sr: int = 22050):
        """Process single file with word-level timestamps"""
        print(f"🎵 Processing: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Step 1: Transcribe
            print("   Transcribing...")
            audio = whisperx.load_audio(str(audio_path.absolute()))
            result = self.whisper_model.transcribe(audio, batch_size=self.batch_size)
            
            language = result.get("language", "en")
            segments = result.get("segments", [])
            print(f"   Language: {language}, Segments: {len(segments)}")
            
            if not segments:
                print("   ⚠️ No segments found")
                return []
            
            # Step 2: Align for word-level timestamps
            align_model, metadata = self._get_alignment_model(language)
            
            if align_model is not None:
                print("   Aligning for word-level timestamps...")
                aligned_result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    device=self.device,
                    return_char_alignments=False,
                )
                segments = aligned_result["segments"]
                print(f"   ✅ Alignment complete, {len(segments)} aligned segments")
            else:
                print("   ⚠️ Using sentence-level timestamps (no alignment)")
                segments = result["segments"]
            
            # Step 3: Load audio for splitting
            print("   Loading audio for splitting...")
            audio_data, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
            
            # Step 4: Split into word-level chunks
            print("   Creating word-level chunks...")
            chunks = []
            
            for seg_idx, segment in enumerate(segments):
                # Get words from segment (if aligned) or use segment text (if not aligned)
                words = segment.get("words", [{"word": segment.get("text", ""), "start": segment.get("start", 0), "end": segment.get("end", 1)}])
                
                for word_idx, word_info in enumerate(words):
                    word_text = word_info.get("word", "").strip()
                    start = word_info.get("start", 0)
                    end = word_info.get("end", start + 0.5)
                    
                    if not word_text or len(word_text) < 2:  # Skip very short words
                        continue
                    
                    # Convert to samples
                    start_sample = max(0, int(start * sr))
                    end_sample = min(len(audio_data), int(end * sr))
                    
                    if end_sample <= start_sample:  # Skip invalid ranges
                        continue
                    
                    # Extract chunk
                    chunk = audio_data[start_sample:end_sample]
                    
                    if len(chunk) < sr * 0.1:  # Skip chunks shorter than 0.1 seconds
                        continue
                    
                    # Generate unique ID
                    clean_name = re.sub(r"[^a-z0-9]", "_", audio_path.name.lower())
                    chunk_id = f"{clean_name}_seg{seg_idx:03d}_word{word_idx:03d}"
                    
                    # Save audio chunk
                    audio_out_path = out_dir / f"{chunk_id}.wav"
                    sf.write(data=chunk, samplerate=sr, file=str(audio_out_path))
                    
                    # Create processed chunk
                    processed = ProcessedChunk(
                        original_audio_path=audio_path,
                        chunk_index=len(chunks),
                        text=word_text,
                        audio_path=audio_out_path,
                        text_path=out_dir / "text",
                        start_time=start,
                        end_time=end
                    )
                    chunks.append(processed)
            
            # Cleanup
            del audio, result, audio_data
            if 'aligned_result' in locals():
                del aligned_result
            cleanup_memory()
            
            elapsed = time.time() - start_time
            print(f"   ✅ Created {len(chunks)} word-level chunks in {elapsed:.1f}s")
            
            return chunks
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            cleanup_memory()
            return []
    
    def write_output_files(self, out_dir: Path, all_chunks: List[ProcessedChunk]):
        """Write output files with word-level information"""
        if not all_chunks:
            return
            
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Write text file (word ID -> word text)
        text_path = out_dir / "text"
        with open(text_path, "w", encoding='utf-8') as f:
            for chunk in all_chunks:
                clean_name = re.sub(r"[^a-z0-9]", "_", chunk.original_audio_path.name.lower())
                chunk_id = f"{clean_name}_seg{chunk.chunk_index//1000:03d}_word{chunk.chunk_index%1000:03d}"
                f.write(f"{chunk_id} {chunk.text}\n")
        
        # Write wav.scp file (word ID -> audio path)
        wav_scp_path = out_dir / "wav.scp"
        with open(wav_scp_path, "w", encoding='utf-8') as f:
            for chunk in all_chunks:
                clean_name = re.sub(r"[^a-z0-9]", "_", chunk.original_audio_path.name.lower())
                chunk_id = f"{clean_name}_seg{chunk.chunk_index//1000:03d}_word{chunk.chunk_index%1000:03d}"
                f.write(f"{chunk_id} {chunk.audio_path.absolute()}\n")
        
        # Write timing file (word ID -> start_time end_time)
        timing_path = out_dir / "timing"
        with open(timing_path, "w", encoding='utf-8') as f:
            for chunk in all_chunks:
                clean_name = re.sub(r"[^a-z0-9]", "_", chunk.original_audio_path.name.lower())
                chunk_id = f"{clean_name}_seg{chunk.chunk_index//1000:03d}_word{chunk.chunk_index%1000:03d}"
                f.write(f"{chunk_id} {chunk.start_time:.3f} {chunk.end_time:.3f}\n")
        
        print(f"✅ Output files written: {len(all_chunks)} chunks")
        print(f"   📝 {text_path}")
        print(f"   🎵 {wav_scp_path}")
        print(f"   ⏱️ {timing_path}")
    
    def get_progress_info(self, out_path: Path):
        """Get processing progress"""
        progress_file = out_path / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
        return {"processed_files": [], "total_chunks": 0}
    
    def save_progress_info(self, out_path: Path, progress_info):
        """Save processing progress"""
        progress_file = out_path / "progress.json"
        out_path.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(progress_info, f, default=str, indent=2)
    
    def process_dataset_batched(self, in_path: Path, out_path: Path, batch_size: int = 25):
        """Process full dataset in batches with proper GPU utilization"""
        wave_paths = list(in_path.glob("*.wav"))
        
        print(f"📊 Dataset Processing Started")
        print(f"   Files found: {len(wave_paths)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Using: {self.device} ({self.model_size} model)")
        
        # Get progress
        progress_info = self.get_progress_info(out_path)
        processed_files = set(progress_info["processed_files"])
        remaining_files = [f for f in wave_paths if str(f) not in processed_files]
        
        if not remaining_files:
            print("✅ All files already processed!")
            return
        
        print(f"   Already processed: {len(processed_files)}")
        print(f"   Remaining: {len(remaining_files)}")
        
        all_chunks = []
        total_processed = len(processed_files)
        
        # Process in batches
        for i in range(0, len(remaining_files), batch_size):
            batch_files = remaining_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(remaining_files) + batch_size - 1) // batch_size
            
            print(f"\n📦 BATCH {batch_num}/{total_batches}")
            batch_start_time = time.time()
            
            batch_chunks = []
            for j, audio_path in enumerate(batch_files):
                print(f"   [{j+1}/{len(batch_files)}] {audio_path.name}")
                
                chunks = self.process_single_file(audio_path, out_path)
                if chunks:
                    batch_chunks.extend(chunks)
                    progress_info["processed_files"].append(str(audio_path))
                    total_processed += 1
                
                # Memory check every 5 files
                if (j + 1) % 5 == 0:
                    memory_usage = check_memory()
                    if memory_usage > 12:  # T4 has ~15GB, leave some buffer
                        print("   🧹 High memory usage, cleaning up...")
                        cleanup_memory()
            
            # Update progress
            all_chunks.extend(batch_chunks)
            progress_info["total_chunks"] = len(all_chunks)
            self.save_progress_info(out_path, progress_info)
            
            # Write output files
            if all_chunks:
                self.write_output_files(out_path, all_chunks)
            
            batch_time = time.time() - batch_start_time
            print(f"   ✅ Batch {batch_num} complete: {len(batch_chunks)} chunks in {batch_time:.1f}s")
            print(f"   📈 Progress: {total_processed}/{len(wave_paths)} files ({total_processed/len(wave_paths)*100:.1f}%)")
            
            # Memory cleanup between batches
            cleanup_memory()
        
        print(f"\n🎉 DATASET PROCESSING COMPLETE!")
        print(f"   Files processed: {total_processed}/{len(wave_paths)}")
        print(f"   Total word-level chunks: {len(all_chunks)}")

