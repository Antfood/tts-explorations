from pathlib import Path
import whisperx
import torch
import librosa
import soundfile as sf
import os
import re
from typing import List
from dataclasses import dataclass
from . import number_utils as nu
from . import constants as const


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
    ):
        self.in_path = in_path
        self.out_path = out_path
        self.metadata_path = metadata_path
        self.language = language
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_compute = compute_type
        self.whisper_model = whisperx.load_model(
            model_size,
            device=self.device,
            language=self.language,
            compute_type=self.whisper_compute,
        )

        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )

    def set_language(self, language: str):

        if self.language == language:
            print(f"Already using {language}. Skip reload.")
            return

        self.language = language
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )

    def transcribe(self, audio_path: Path):
        audio = whisperx.load_audio(audio_path.absolute())
        return audio, self.whisper_model.transcribe(audio, batch_size=self.batch_size)

    def align(self, audio, transcription) -> dict:
        if self.language != transcription["language"]:
            print(":: align model language does not match transcription.")
            print(
                f':: Align Model: {self.language} | transcription {transcription["language"]}'
            )
            print(":: Reloading Align model with transcription language.")
            self.set_language(transcription["language"])

        return whisperx.align(
            transcription["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            device=self.device,
            return_char_alignments=False,
        )

    def split_audio(
        self,
        audio_path: Path,
        aligned_result: dict,
        target_sr: int = 24000,
    ) -> List[ProcessedChunk]:

        audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        chunks = []

        for i, segment in enumerate(aligned_result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = nu.normalize_text_number(segment["text"].strip(), self.language)

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            chunk = audio_data.data[start_sample:end_sample]

            clean_name = clean_name = re.sub(r"[^a-z0-9]", "_", audio_path.name.lower())
            id = f"{clean_name}_chunk-{i:04d}"

            audio_out_path = self.out_path / f"{id}.wav"
            text_out_path = self.metadata_path / "text"
            wav_scp_path = self.metadata_path / "wav.scp"
           

            sf.write(data=chunk, samplerate=sr, file=audio_out_path)

            with open(text_out_path, "a") as f:
                f.write(f"{id} {text}\n")

            with open(wav_scp_path, "a") as f:
                f.write(f"{id} {audio_out_path.relative_to(self.out_path)}\n")

            processed = ProcessedChunk(
                original_audio_path=audio_path,
                chunk_index=i,
                text=text,
                audio_path=audio_out_path,
                text_path=text_out_path,
            )

            chunks.append(processed)

        return chunks

    def preprocess(self):
        wave_paths = list(self.in_path.glob("*.wav"))
    
        print(f":: Found {len(wave_paths)} audio files in {self.in_path}.")
        print(f":: Processing files to {self.out_path}.")
    
        for i, audio_path in enumerate(wave_paths):
            print(f":: Processing file {i + 1}/{len(wave_paths)}: {audio_path.name}")
            audio, transcription = self.transcribe(audio_path)
    
            aligned_results = self.align(audio, transcription)
            self.out_path.mkdir(parents=True, exist_ok=True)
    
            yield self.split_audio(
                audio_path,
                aligned_results,
                target_sr=22050,
            )

    def init_files(self):
        text_path = self.metadata_path / "text"
        wav_scp_path = self.metadata_path / "wav.scp"
    
        text_path.write_text("")
        wav_scp_path.write_text("")
    
    


