import io
import csv
import os
import time
import warnings
from dataclasses import asdict, dataclass, fields
from itertools import islice
from pathlib import Path

import sentencepiece
import torch
import sphn
from hibiki_zero.client_utils import audio_read, stack_and_pad_audio
from hibiki_zero.run import DEFAULT_REPO

from moshi.models import LMGen, MimiModel, loaders
from hibiki_zero.inference import get_lmgen, seed_all, encode_inputs

# Use 10 for the initial proof-of-concept. Set this to None for the full dataset.
dataset_row_limit: int | None = 10


def get_vsc5_paths() -> tuple[Path, Path, Path]:
    data_environment_variable = os.environ.get("DATA")
    if not data_environment_variable:
        raise RuntimeError(
            "The DATA environment variable is not set"
        )

    data_directory = Path(data_environment_variable).expanduser().resolve()
    dataset_directory = (
        data_directory
        / "24kHz covost_v2-de_en"
        / "covost_v2-de_en"
    )
    model_results_directory = (
        data_directory
        / "unchecked_model_results"
        / "hibiki_zero"
    )

    return (
        dataset_directory / "covost_v2.de_en.test.tsv",
        model_results_directory / "output_results.tsv",
        model_results_directory / "translated_audio",
    )

# Since Hibiki-Zero doesn't directly support big dataset testing (is limited to batch functionality) it is needed to load
# the model components so that translation tests can be run in a loop without always restarting/reloading the model when
# using the standard hibiki-zero generate command
# https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L35
@dataclass
class LoadedModel:
    mimi_instance: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen


@dataclass(frozen=True)
class GeneratedTranslation:
    translated_text: str
    path_to_translated_audio: Path
    # Time that passed since the start of the input audio till the first chunk of the output audio (first chunk could be just silence)
    output_audio_start_delay: float

@dataclass
class ModelResult:
    path_to_original_audio: Path
    original_transcription: str
    expected_translation_of_original_audio: str
    translated_text: str
    path_to_translated_audio: Path
    # Time that passed since the start of the input audio till the first chunk of the output audio (maybe silence)
    output_audio_start_delay: float

    # The following fields are being filled upon post-processing the performance/quality
    # Time that passed from the beginning of the first input word till the beginning of the first output word
    # float | None is seen as an exclusive "OR" typification
    ttfa: float | None = None
    # Time that passed from the end of the last input word till the end of the last output word
    end_offset: float | None = None
    bleu: float | None = None
    asr_bleu: float | None = None
    asr_comet: float | None = None

def main() -> None:
    (
        dataset_path,
        output_results_path,
        translated_audio_output_directory,
    ) = get_vsc5_paths()

    dataset_audio_path_transcription_and_translation_data = loading_dataset(
        dataset_path=dataset_path,
        row_limit=dataset_row_limit,
    )
    model = loading_model()
    run_model(
        dataset_audio_path_transcription_and_translation_data,
        model,
        output_results_path=output_results_path,
        translated_audio_output_directory=translated_audio_output_directory,
    )


# Step one: Making the Dataset Accessible:
def loading_dataset(
        dataset_path: Path,
        row_limit: int | None = None,
) -> list[dict[str, str]]:
    with io.open(dataset_path, "r", encoding="utf-8", newline="") as dataset_tsv_file:
        # DictReader leaves out the fieldnames (1st row) https://docs.python.org/3/library/csv.html#csv.DictReader
        reader = csv.DictReader(dataset_tsv_file, delimiter="\t")
        # Only let's part of the dataset being translated if row_limit is not None
        rows = list(reader if row_limit is None else islice(reader, row_limit))

    # Rewrites the relative audio path into the absolute audio path of this model.
    for row in rows:
        audio_path = Path(row["path"])
        if not audio_path.is_absolute():
            audio_path = dataset_path.parent / audio_path
        row["path"] = str(audio_path.resolve())

    return rows


# Step two: Loading the model (Method was build by analyzing the hibiki-zero gitlab repo (see links below)).
def loading_model(device: str = "cuda", fuse_lora=True, ) -> LoadedModel:
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L168 last visit: 12.10.2026
    seed_all(42)

    hf_repo_parts = DEFAULT_REPO.split("@")
    hf_repo_name = hf_repo_parts[0]
    revision = hf_repo_parts[1]

    # Contains paths to checkpoint data (LLM-Weights)/sub models
    # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L148 last visit: 12.10.2026
    # ---
    # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L181 last visit: 12.10.2026
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L236 last visit: 12.10.2026
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=hf_repo_name,
        revision=revision,
    )

    # Loads the mimi model.
    # Mimi converts audio into audio tokens and vice versa
    # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L282 last visit: 12.10.2026
    # ---
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L247 last visit: 12.10.2026
    mimi = checkpoint_info.get_mimi(device=device)

    # Loads the text tokenizer of Google's SentencePiece
    # Converts text into token IDs and vice versa
    # Example from Codex:
    # Text:      "Hello world"
    # Tokens:    ["▁Hello", "▁world"]
    # Token-IDs: [1847, 932]
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L248 last visit: 12.10.2026
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    # Hibiki-Zero built's up on the architecture of moshi -> When calling get_moshi the moshi_weights variable points to the
    # weights of the Hibiki-Zero model (This is due to calling loaders.CheckpointInfo.from_hf_repo with the repo from hibiki a few rows above)
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L251 last visit: 12.10.2026
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16, fuse_lora=fuse_lora)

    # Generates text/audio tokens based on the previous tokens. (lm calculates the predicted next tokens)
    # https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L267 last visit: 12.10.2026
    lm_gen: LMGen = get_lmgen(lm=lm, checkpoint_info=checkpoint_info, batch_size=1)

    return LoadedModel(
        mimi_instance=mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
    )


# Step three: Translating Audio Files into Text and Audio
def run_model(dataset: list[dict[str, str]],
              loaded_model: LoadedModel,
              output_results_path: Path,
              translated_audio_output_directory: Path,
              ) -> list[ModelResult]:

    model_results: list[ModelResult] = []

    translated_audio_output_directory.mkdir(parents=True, exist_ok=True)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)

    #Defines the needed columns of the output result tsv
    fieldnames = [field.name for field in fields(ModelResult)]

    with output_results_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row_data in dataset:
            #Writes the already known data into the variables that are used to built up the result tsv
            path_to_original_audio = Path(row_data["path"]).resolve()
            original_transcription = row_data["sentence"]
            expected_translation = row_data["translation"]

            #Reuses the unique identifier of the original audio paths
            output_audio_path = (
                    translated_audio_output_directory
                    / f"{path_to_original_audio.stem}.wav"
            )

            #Hibiki translation
            generated_translation = generate_translation(
                path_to_original_audio=path_to_original_audio,
                output_audio_path=output_audio_path,
                loaded_model=loaded_model,
            )

            model_result = ModelResult(
                path_to_original_audio=path_to_original_audio,
                original_transcription=original_transcription,
                expected_translation_of_original_audio=expected_translation,
                translated_text=generated_translation.translated_text,
                path_to_translated_audio=generated_translation.path_to_translated_audio.resolve(),
                output_audio_start_delay=generated_translation.output_audio_start_delay,
            )
            model_results.append(model_result)
            writer.writerow(asdict(model_result))
            output_file.flush()

    return model_results

#TODO read about torch.no_grad functionality

@torch.no_grad()
def generate_translation(path_to_original_audio: Path,
                         output_audio_path: Path,
                         loaded_model: LoadedModel,
                         generation_window_seconds: float = 120.0,
                         ) -> GeneratedTranslation:
    # mimi = loaded_model.mimi_instance
    # lm_gen = loaded_model.lm_gen
    # text_tokenizer = loaded_model.text_tokenizer
    # sample_rate = mimi.sample_rate

    # original_audio, _ = audio_read(
    #    path_to_original_audio,
    #    mono=True,
    # )
    # start_of_input_stream = time.perf_counter()

    # tranlsated_text = text_tokenizer.decode(
    #    translated_text_tokens
    # )

    # return GeneratedTranslation(
    #    translated_text=translated_text,
    #    path_to_translated_audio=path_to_translated_audio,
    #    output_audio_start_delay=output_audio_start_delay
    # )

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # Just copy pasted for a Proof of Concept
    # TODO: Code review/revision/description
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    """
        Generates one translation with batch size 1.

        Input tokens are supplied according to Mimi's real-time frame rate.
        The model weights remain loaded between calls; only the streaming
        state is recreated.
        """
    mimi = loaded_model.mimi_instance
    lm_gen = loaded_model.lm_gen
    text_tokenizer = loaded_model.text_tokenizer

    sample_rate = mimi.sample_rate
    frame_duration_seconds = 1.0 / mimi.frame_rate

    # The data is guaranteed to be WAV at 24 kHz.
    input_wav, _ = audio_read(
        path_to_original_audio,
        mono=True,
    )

    # Ensures shape [channels, samples].
    # This is shape normalization, not a format/sample-rate check.
    if input_wav.ndim == 1:
        input_wav = input_wav.unsqueeze(0)

    input_duration_seconds = (
            input_wav.shape[-1] / sample_rate
    )

    # Batch size is exactly 1.
    batch_wav = stack_and_pad_audio(
        [input_wav],
        max_len=int(
            generation_window_seconds * sample_rate
        ),
    )

    # Encoding is performed before the streaming clock begins.
    # During generation, the resulting codes are supplied at 12.5 Hz.
    input_codes, warmup_codes = encode_inputs(
        batch_wavs=batch_wav,
        mimi=mimi,
        lm_gen=lm_gen,
        audio_durations=[input_duration_seconds],
    )

    generated_text_token_ids: list[int] = []

    # Contains decoded audio and any silence inserted for runtime gaps.
    output_audio_parts: list[torch.Tensor] = []

    output_audio_start_seconds: float | None = None
    current_output_duration_seconds = 0.0
    eos_generated = False
    parts_after_last_text_token: int | None = None

    device = lm_gen.lm_model.device

    # Both objects receive a fresh streaming state for this file.
    with lm_gen.streaming(batch_size=1), mimi.streaming(
            batch_size=1
    ):
        # Algorithmic preparation used by the official generate()
        # implementation. It is completed before the measurement starts.
        for step in range(warmup_codes.shape[-1]):
            lm_gen.step(
                warmup_codes[:, :, step: step + 1]
            )

        # Ensure that preceding CUDA work is complete before starting
        # the common input/output clock.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        input_stream_started_at = time.perf_counter()

        for step in range(input_codes.shape[-1]):
            # Input code step 0 represents the first complete Mimi frame.
            # It becomes available after one frame duration.
            frame_deadline = (
                    input_stream_started_at
                    + (step + 1) * frame_duration_seconds
            )

            remaining_wait_seconds = (
                    frame_deadline - time.perf_counter()
            )

            if remaining_wait_seconds > 0:
                time.sleep(remaining_wait_seconds)

            tokens = lm_gen.step(
                input_codes[:, :, step: step + 1]
            )

            if tokens is None:
                continue

            text_token_id = int(
                tokens[0, 0, 0].item()
            )

            source_timeline_seconds = (
                    (step + 1) * frame_duration_seconds
            )

            # Ignore an EOS emitted before the source has finished.
            # This reflects Hibiki's intention to avoid premature EOS.
            if (
                    text_token_id == text_tokenizer.eos_id()
                    and source_timeline_seconds
                    < input_duration_seconds
            ):
                text_token_id = text_tokenizer.pad_id()

            generated_text_token_ids.append(
                text_token_id
            )

            # EOS marks the end of the target sequence. The audio tokens
            # belonging to this EOS step are not included in the output.
            if text_token_id == text_tokenizer.eos_id():
                eos_generated = True
                break

            # Channel 0 is text; channels 1... contain target audio.
            decoded_chunk = mimi.decode(
                tokens[:, 1:]
            )

            # Moving to CPU waits until the decoded PCM is actually ready.
            decoded_chunk = decoded_chunk.cpu()[0]

            chunk_ready_seconds = (
                    time.perf_counter()
                    - input_stream_started_at
            )

            (
                output_audio_start_seconds,
                current_output_duration_seconds,
            ) = append_audio_chunk_to_timeline(
                audio_parts=output_audio_parts,
                audio_chunk=decoded_chunk,
                chunk_ready_seconds=chunk_ready_seconds,
                output_audio_start_seconds=(
                    output_audio_start_seconds
                ),
                current_output_duration_seconds=(
                    current_output_duration_seconds
                ),
                sample_rate=sample_rate,
            )

            if text_token_id > text_tokenizer.pad_id():
                parts_after_last_text_token = len(output_audio_parts)

    if output_audio_start_seconds is None:
        raise RuntimeError(
            f"Hibiki produced no audio for {path_to_original_audio}."
        )

    if not eos_generated:
        warnings.warn(
            f"Hibiki produced no output EOS token for "
            f"{path_to_original_audio}. The output is truncated after the "
            f"last generated text token.",
            RuntimeWarning,
        )
        if parts_after_last_text_token is None:
            raise RuntimeError(
                f"Hibiki produced neither an EOS token nor a non-padding "
                f"text token for {path_to_original_audio}."
            )
        output_audio_parts = output_audio_parts[:parts_after_last_text_token]

    # Remove PAD, BOS and other reserved tokens from the text.
    translated_text_token_ids = [
        token_id
        for token_id in generated_text_token_ids
        if token_id > text_tokenizer.pad_id()
           and token_id != text_tokenizer.eos_id()
    ]

    translated_text = text_tokenizer.decode(
        translated_text_token_ids
    )

    output_wav = torch.cat(
        output_audio_parts,
        dim=-1,
    )

    output_audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sphn.write_wav(
        output_audio_path,
        output_wav.numpy(),
        sample_rate,
    )

    return GeneratedTranslation(
        translated_text=translated_text,
        path_to_translated_audio=output_audio_path,
        output_audio_start_delay=(
            output_audio_start_seconds
        ),
    )

    ########################################################################################################
    ########################################################################################################


    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # Just copy pasted for a Proof of Concept
    # TODO: Code review/revision/description
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
def append_audio_chunk_to_timeline(
        audio_parts: list[torch.Tensor],
        audio_chunk: torch.Tensor,
        chunk_ready_seconds: float,
        output_audio_start_seconds: float | None,
        current_output_duration_seconds: float,
        sample_rate: int,
) -> tuple[float, float]:
    """
    Adds a decoded audio chunk to the reconstructed playback timeline.

    If a chunk becomes available too late for uninterrupted playback,
    silence is inserted. Therefore, the saved WAV also represents
    generation stalls that would be audible during real-time playback.

    Returns:
        output_audio_start_seconds
        updated current_output_duration_seconds
    """
    chunk_duration_seconds = (
            audio_chunk.shape[-1] / sample_rate
    )

    if output_audio_start_seconds is None:
        output_audio_start_seconds = chunk_ready_seconds
        audio_parts.append(audio_chunk)

        return (
            output_audio_start_seconds,
            chunk_duration_seconds,
        )

    # Position at which this chunk becomes available, measured
    # relative to the start of the output WAV.
    chunk_available_in_output_seconds = (
            chunk_ready_seconds
            - output_audio_start_seconds
    )

    # If playback has already reached the current end of the file
    # before the next chunk is ready, this difference is an audible gap.
    gap_seconds = (
            chunk_available_in_output_seconds
            - current_output_duration_seconds
    )

    if gap_seconds > 0:
        gap_samples = round(gap_seconds * sample_rate)

        silence = torch.zeros(
            audio_chunk.shape[0],
            gap_samples,
            dtype=audio_chunk.dtype,
        )

        audio_parts.append(silence)

        current_output_duration_seconds += (
                gap_samples / sample_rate
        )

    audio_parts.append(audio_chunk)

    current_output_duration_seconds += (
        chunk_duration_seconds
    )

    return (
        output_audio_start_seconds,
        current_output_duration_seconds,
    )
    ########################################################################################################
    ########################################################################################################



if __name__ == "__main__":
    main()
