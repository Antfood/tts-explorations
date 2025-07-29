from pathlib import Path
import whisperx
import torch
import librosa
import soundfile as sf
import os
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


DEFAULT_LANGUAGE = "pt"
WHISPER_BATCH = 8

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

device = "cpu"
language = DEFAULT_LANGUAGE
whisper_compute = "float32"

if torch.cuda.is_available():
    print(":: DEVICE: USING CUDA")
    device = "cuda"
else:
    print(":: DEVICE: USING CPU")

whisper_model = whisperx.load_model(
    "small", device=device, language=DEFAULT_LANGUAGE, compute_type=whisper_compute
)


def load_align_model(language: str):
    return whisperx.load_align_model(
        language_code=language,
        device=device,
    )


align_model, metadata = load_align_model(DEFAULT_LANGUAGE)


def set_language(lan):
    global align_model
    global metadata
    global language

    if language == lan:
        print(f"Already using {lan}. Skip reload.")
        return

    language = lan
    align_model, metadata = load_align_model(language)


def transcribe(audio_path: Path, batch_size: int = WHISPER_BATCH):
    audio = whisperx.load_audio(audio_path.absolute())
    return audio, whisper_model.transcribe(audio, batch_size=batch_size)


def align(audio, transcription) -> dict:
    if language != transcription["language"]:
        print(":: align model language does not match transcription.")
        print(f':: Align Model: {language} | transcription {transcription["language"]}')
        print(":: Reloading Align model with transcription language.")
        set_language(transcription["language"])

    return whisperx.align(
        transcription["segments"],
        align_model,
        metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )


def split_audio(
    audio_path: Path, out_dir: Path, aligned_result: dict
) -> List[ProcessedChunk]:
    audio_data, sr = librosa.load(audio_path, sr=None)
    written = []

    for i, segment in enumerate(aligned_result["segments"]):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        chunk = audio_data.data[start_sample:end_sample]
        audio_out_path = out_dir / f"{audio_path.name}_chunk-{i:04d}.wav"
        text_out_path = out_dir / f"{audio_path.name}_chunk-{i:04d}.txt"

        sf.write(data=chunk, samplerate=sr, file=audio_out_path)

        with open(text_out_path, "w+") as f:
            f.write(text)

        processed = ProcessedChunk(
            original_audio_path=audio_path,
            chunk_index=i,
            text=text,
            audio_path=audio_out_path,
            text_path=text_out_path,
        )

        written.append(processed)

    return written
