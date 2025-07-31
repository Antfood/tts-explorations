from pathlib import Path
import whisperx
import torch
import librosa
import soundfile as sf
import os
import re
import gc
import json
from typing import List
from dataclasses import dataclass

@dataclass
class ProcessedChunk:
    original_audio_path: Path
    text_path: Path
    audio_path: Path
    text: str
    chunk_index: int

    def to_list(self):
        return [self.original_audio_path, self.text_path, self.audio_path, self.text, self.chunk_index]

def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_memory():
    """Check current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("   Using CPU")

class StepByStepProcessor:
    def __init__(self):
        self.whisper_model = None
        self.align_model = None
        self.metadata = None
        self.device = "cpu"  # Start with CPU to avoid crashes
        self.whisper_compute = "float32"
        
    def step1_load_whisper_model(self):
        """Step 1: Load Whisper model"""
        print("🔄 STEP 1: Loading Whisper model...")
        try:
            self.whisper_model = whisperx.load_model(
                "tiny",  # Use tiny model first
                self.device, 
                compute_type=self.whisper_compute
            )
            print("✅ Step 1 SUCCESS: Whisper model loaded")
            check_memory()
            return True
        except Exception as e:
            print(f"❌ Step 1 FAILED: {e}")
            return False
    
    def step2_transcribe_audio(self, audio_path: Path):
        """Step 2: Transcribe audio"""
        print("🔄 STEP 2: Transcribing audio...")
        try:
            audio = whisperx.load_audio(str(audio_path.absolute()))
            print(f"   Audio loaded: {len(audio)} samples")
            
            result = self.whisper_model.transcribe(audio, batch_size=1)
            print(f"   Language detected: {result.get('language', 'unknown')}")
            print(f"   Segments found: {len(result.get('segments', []))}")
            
            print("✅ Step 2 SUCCESS: Audio transcribed")
            check_memory()
            return audio, result
        except Exception as e:
            print(f"❌ Step 2 FAILED: {e}")
            return None, None
    
    def step3_load_align_model(self, language: str):
        """Step 3: Load alignment model - THIS IS LIKELY WHERE IT CRASHES"""
        print("🔄 STEP 3: Loading alignment model...")
        print(f"   Loading alignment model for language: {language}")
        try:
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            print("✅ Step 3 SUCCESS: Alignment model loaded")
            check_memory()
            return True
        except Exception as e:
            print(f"❌ Step 3 FAILED: {e}")
            print("   This is likely where your original code crashes!")
            return False
    
    def step4_align_transcription(self, audio, transcription):
        """Step 4: Align transcription"""
        print("🔄 STEP 4: Aligning transcription...")
        try:
            aligned_result = whisperx.align(
                transcription["segments"],
                self.align_model,
                self.metadata,
                audio,
                device=self.device,
                return_char_alignments=False,
            )
            print("✅ Step 4 SUCCESS: Transcription aligned")
            check_memory()
            return aligned_result
        except Exception as e:
            print(f"❌ Step 4 FAILED: {e}")
            return None
    
    def step5_split_audio(self, audio_path: Path, out_dir: Path, aligned_result: dict):
        """Step 5: Split audio into chunks"""
        print("🔄 STEP 5: Splitting audio...")
        try:
            # Load audio with librosa
            audio_data, sr = librosa.load(str(audio_path), sr=22050, mono=True)
            print(f"   Audio data loaded: {len(audio_data)} samples at {sr}Hz")
            
            written = []
            segments = aligned_result["segments"]
            print(f"   Processing {len(segments)} segments...")
            
            for i, segment in enumerate(segments):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                chunk = audio_data[start_sample:end_sample]
                
                clean_name = re.sub(r"[^a-z0-9]", "_", audio_path.name.lower())
                id = f"{clean_name}_chunk-{i:04d}"
                
                audio_out_path = out_dir / f"{id}.wav"
                
                # Write audio chunk
                sf.write(data=chunk, samplerate=sr, file=str(audio_out_path))
                
                processed = ProcessedChunk(
                    original_audio_path=audio_path,
                    chunk_index=i,
                    text=text,
                    audio_path=audio_out_path,
                    text_path=out_dir / "text",
                )
                written.append(processed)
                
                # Progress update every 10 chunks
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(segments)} chunks")
            
            print("✅ Step 5 SUCCESS: Audio split into chunks")
            check_memory()
            return written
            
        except Exception as e:
            print(f"❌ Step 5 FAILED: {e}")
            return []
    
    def step6_write_output_files(self, out_dir: Path, chunks: List[ProcessedChunk]):
        """Step 6: Write text and wav.scp files"""
        print("🔄 STEP 6: Writing output files...")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            
            text_path = out_dir / "text"
            wav_scp_path = out_dir / "wav.scp"
            
            # Write text file
            with open(text_path, "w", encoding='utf-8') as f:
                for chunk in chunks:
                    clean_name = re.sub(r"[^a-z0-9]", "_", chunk.original_audio_path.name.lower())
                    id = f"{clean_name}_chunk-{chunk.chunk_index:04d}"
                    f.write(f"{id} {chunk.text}\n")
            
            # Write wav.scp file
            with open(wav_scp_path, "w", encoding='utf-8') as f:
                for chunk in chunks:
                    clean_name = re.sub(r"[^a-z0-9]", "_", chunk.original_audio_path.name.lower())
                    id = f"{clean_name}_chunk-{chunk.chunk_index:04d}"
                    f.write(f"{id} {chunk.audio_path.absolute()}\n")
            
            print(f"✅ Step 6 SUCCESS: Output files written")
            print(f"   Text file: {text_path}")
            print(f"   Wav.scp file: {wav_scp_path}")
            print(f"   Total chunks: {len(chunks)}")
            return True
            
        except Exception as e:
            print(f"❌ Step 6 FAILED: {e}")
            return False
    
    def process_single_file_debug(self, audio_path: Path, out_dir: Path):
        """Process single file with detailed step-by-step debugging"""
        print("🚀 STARTING STEP-BY-STEP PROCESSING")
        print(f"File: {audio_path.name}")
        print("=" * 60)
        
        # Step 1: Load Whisper model
        if not self.step1_load_whisper_model():
            return False
        
        # Step 2: Transcribe
        audio, transcription = self.step2_transcribe_audio(audio_path)
        if audio is None or transcription is None:
            return False
        
        # Step 3: Load alignment model (LIKELY CRASH POINT)
        language = transcription.get("language", "en")
        if not self.step3_load_align_model(language):
            print("⚠️  ALIGNMENT MODEL FAILED - This is likely your crash point!")
            print("   Trying to continue without alignment...")
            # Use original transcription without alignment
            aligned_result = transcription
        else:
            # Step 4: Align transcription
            aligned_result = self.step4_align_transcription(audio, transcription)
            if aligned_result is None:
                print("   Using transcription without alignment...")
                aligned_result = transcription
        
        # Step 5: Split audio
        chunks = self.step5_split_audio(audio_path, out_dir, aligned_result)
        if not chunks:
            return False
        
        # Step 6: Write output files
        if not self.step6_write_output_files(out_dir, chunks):
            return False
        
        # Cleanup
        del audio, transcription, aligned_result, chunks
        cleanup_memory()
        
        print("🎉 PROCESSING COMPLETE!")
        return True

# Create processor and test on your problem file
processor = StepByStepProcessor()

# Test on the problematic file
success = processor.process_single_file_debug(
    Path("/content/drive/MyDrive/Machine Learning/data/lou_vo/Lou_VO_Vichy_240229_Take2.wav"),
    Path("/content/drive/MyDrive/test_output")
)

if success:
    print("\n✅ SUCCESS! Now we know the process works.")
    print("You can use this processor for all your files.")
else:
    print("\n❌ We found where it crashes. Check the step that failed above.")
