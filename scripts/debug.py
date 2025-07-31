import torch
import whisperx
import librosa
import gc
from pathlib import Path
import traceback


def check_system_info():
    """Check system resources and CUDA setup"""
    print("=== SYSTEM INFO ===")

    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"GPU Memory - Total: {total_memory:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )
    else:
        print("CUDA Available: False")

    # Check cuDNN
    print(f"cuDNN Available: {torch.backends.cudnn.is_available()}")
    print(
        f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}"
    )

    print("==================")


def test_audio_file(audio_path):
    """Test if audio file can be loaded"""
    print(f"\n=== TESTING AUDIO FILE: {audio_path.name} ===")

    # Check file exists and size
    if not audio_path.exists():
        print(f"ERROR: File does not exist: {audio_path}")
        return False

    file_size = audio_path.stat().st_size / 1024**2  # MB
    print(f"File size: {file_size:.2f} MB")

    if file_size > 500:  # Very large file
        print(f"WARNING: File is very large ({file_size:.2f} MB)")

    # Test with librosa
    try:
        print("Testing librosa load...")
        audio_data, sr = librosa.load(
            str(audio_path), sr=None, duration=10
        )  # Load only first 10 seconds
        print(
            f"✓ Librosa load successful - Duration: {len(audio_data)/sr:.2f}s, Sample rate: {sr}Hz"
        )
        del audio_data  # Clean up
        gc.collect()
        return True
    except Exception as e:
        print(f"✗ Librosa load failed: {e}")
        return False


def test_whisperx_load(audio_path):
    """Test WhisperX audio loading"""
    print(f"\n=== TESTING WHISPERX LOAD ===")
    try:
        print("Loading audio with WhisperX...")
        audio = whisperx.load_audio(str(audio_path))
        print(f"✓ WhisperX load successful - Audio shape: {audio.shape}")
        del audio
        gc.collect()
        return True
    except Exception as e:
        print(f"✗ WhisperX load failed: {e}")
        print(f"Full error: {traceback.format_exc()}")
        return False


def test_model_loading():
    """Test model loading step by step"""
    print(f"\n=== TESTING MODEL LOADING ===")

    # Test different configurations
    configs = [
        {"model": "tiny", "device": "cpu", "compute_type": "float32"},
        (
            {"model": "tiny", "device": "cuda", "compute_type": "float16"}
            if torch.cuda.is_available()
            else None
        ),
        {"model": "small", "device": "cpu", "compute_type": "float32"},
    ]

    configs = [c for c in configs if c is not None]

    for i, config in enumerate(configs):
        try:
            print(f"\nTesting config {i+1}: {config}")
            model = whisperx.load_model(
                config["model"], config["device"], compute_type=config["compute_type"]
            )
            print(f"✓ Model loaded successfully")
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            return config
        except Exception as e:
            print(f"✗ Config failed: {e}")
            continue

    print("All model configurations failed!")
    return None


def test_transcription_minimal(audio_path):
    """Test minimal transcription"""
    print(f"\n=== TESTING MINIMAL TRANSCRIPTION ===")

    # Find working model config
    working_config = test_model_loading()
    if not working_config:
        print("No working model configuration found!")
        return False

    try:
        print(f"Using working config: {working_config}")

        # Load model
        model = whisperx.load_model(
            working_config["model"],
            working_config["device"],
            compute_type=working_config["compute_type"],
        )

        # Load audio (only first 30 seconds)
        print("Loading audio (30s max)...")
        audio = whisperx.load_audio(str(audio_path))
        if len(audio) > 30 * 16000:  # Trim to 30 seconds
            audio = audio[: 30 * 16000]
            print("Trimmed to 30 seconds")

        # Transcribe with minimal batch size
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=1)

        print(f"✓ Transcription successful!")
        print(f"Language detected: {result.get('language', 'unknown')}")
        print(f"Segments found: {len(result.get('segments', []))}")

        if result.get("segments"):
            first_segment = result["segments"][0]
            print(f"First segment: '{first_segment.get('text', '')[:50]}...'")

        # Cleanup
        del model, audio, result
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return True

    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        print(f"Full error: {traceback.format_exc()}")
        return False


def debug_first_file(audio_path):
    """Complete debug of the first file"""
    print("🔍 DEBUGGING FIRST FILE CRASH")
    print("=" * 50)

    # Step 1: System check
    check_system_info()

    # Step 2: File check
    if not test_audio_file(audio_path):
        print("❌ Audio file test failed - file issue")
        return

    # Step 3: WhisperX load test
    if not test_whisperx_load(audio_path):
        print("❌ WhisperX load failed - audio format issue")
        return

    # Step 4: Model loading test
    if not test_model_loading():
        print("❌ Model loading failed - CUDA/model issue")
        return

    # Step 5: Minimal transcription test
    if not test_transcription_minimal(audio_path):
        print("❌ Transcription failed - processing issue")
        return

    print("\n✅ ALL TESTS PASSED!")
    print("The issue might be with:")
    print("1. Processing large batch of files")
    print("2. Memory accumulation")
    print("3. Alignment model loading")
    print("4. File output operations")


# Usage:
# debug_first_file(Path("/content/drive/MyDrive/Machine Learning/data/lou_vo/Lou_VO_Vichy_240229_Take2.wav"))


def create_minimal_processor():
    """Create a minimal processor that should work"""
    print("\n=== CREATING MINIMAL PROCESSOR ===")

    # Find working config
    working_config = test_model_loading()
    if not working_config:
        return None

    try:
        # Load only whisper model, skip alignment for now
        model = whisperx.load_model(
            working_config["model"],
            working_config["device"],
            compute_type=working_config["compute_type"],
        )

        def process_single_file_minimal(audio_path, out_dir):
            """Process single file with minimal operations"""
            try:
                print(f"Processing: {audio_path.name}")

                audio = whisperx.load_audio(str(audio_path))
                result = model.transcribe(audio, batch_size=1)

                out_dir.mkdir(parents=True, exist_ok=True)
                transcript_file = out_dir / f"{audio_path.stem}_transcript.txt"

                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in result.get("segments", []):
                        f.write(f"{segment.get('text', '')}\n")

                print(f"✓ Saved transcript to: {transcript_file}")

                del audio, result
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                return True

            except Exception as e:
                print(f"✗ Failed: {e}")
                return False

        return process_single_file_minimal

    except Exception as e:
        print(f"Failed to create processor: {e}")
        return None
