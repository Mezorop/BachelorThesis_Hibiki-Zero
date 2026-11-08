# Portions of this file are adapted from Hibiki-Zero:
# Copyright (c) 2026 kyutai
# Licensed under the MIT License. See THIRD_PARTY_NOTICES.md and
# LICENSES/Hibiki-Zero-MIT.txt for attribution and the complete license text.
# Source revision: https://github.com/kyutai-labs/hibiki-zero/tree/871e89d078202c7d9d18d0924bd76cf161cd6606

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

from hibiki_zero.client_utils import audio_read
from hibiki_zero.run import DEFAULT_REPO

from moshi.models import LMGen, MimiModel, loaders
from hibiki_zero.inference import get_lmgen, seed_all

# Using 10 rows for the initial proof-of-concept.
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


# Adapted from Hibiki-Zero's model-loading and inference implementation under
# the MIT License; see the file-level notice above.
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

    # Runs the warmup process before starting the live translation
    # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L64
    warm_up_live_model(mimi=mimi, lm_gen=lm_gen)

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

    # Defines the needed columns of the output result tsv
    fieldnames = [field.name for field in fields(ModelResult)]

    with output_results_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row_data in dataset:
            # Writes the already known data into the variables that are used to built up the result tsv
            path_to_original_audio = Path(row_data["path"]).resolve()
            original_transcription = row_data["sentence"]
            expected_translation = row_data["translation"]

            # Reuses the unique identifier of the original audio paths
            output_audio_path = (
                    translated_audio_output_directory
                    / f"{path_to_original_audio.stem}.wav"
            )

            # Hibiki translation
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


# torch.no_grad() is recommended when only running a trained model and not needing information for model training.
# https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html, last visit 21.10.2026
@torch.no_grad()
def warm_up_live_model(mimi: MimiModel, lm_gen: LMGen) -> None:
    device = lm_gen.lm_model.device
    # A model step is 80 milliseconds and now the question is how many samples are included in that time frame
    # This boils down to 24000/12.5 = 1920 audio samples per frame
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    with mimi.streaming(batch_size=1), lm_gen.streaming(batch_size=1):

        # model warmup
        # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L64-74
        for _warmup_step in range(4):
            chunk = torch.zeros(
                1,
                1,
                frame_size,
                dtype=torch.float32,
                device=device,
            )
            # Encoding of input audio returns "... an int tensor of shape [B, K, T] ..." K represents the amount of audio tokens
            # of a single frame with each token representing a different audio detail.
            # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/compression.py#L383
            codes = mimi.encode(chunk)
            # codes is an int tensor of shape [B, K, T] (See variable initialization) and T being the audio frames
            for code_index in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, code_index:code_index + 1])
                # ":" -> all batch entries (in our case always 1)
                # ":" -> all code books
                # "0: 0 + 1" -> all audio frames from position 0 till but not included 1.
                if tokens is not None:
                    _ = mimi.decode(tokens[:, 1:])

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_translation(path_to_original_audio: Path,
                         output_audio_path: Path,
                         loaded_model: LoadedModel,
                         generation_window_seconds: float = 120.0,
                         ) -> GeneratedTranslation:
    """
    Replays one audio file through Hibiki-Zero's live inference path.

    Raw mono PCM is made available one Mimi frame at a time. Each frame is
    encoded by Mimi, processed by LMGen, and any generated target-audio tokens
    are immediately decoded by Mimi. Model weights remain loaded between
    calls, while the streaming state is recreated for every file.

    The finite-file evaluation adds an input-EOS frame after the source audio
    and then streams silence until the target EOS or generation-window limit.
    """

    mimi = loaded_model.mimi_instance
    lm_gen = loaded_model.lm_gen
    text_tokenizer = loaded_model.text_tokenizer

    sample_rate = mimi.sample_rate
    # mimi runs on a framerate of 12.5Hz
    # Therefore 1/12.5 = 0.08s means that the model steps are 80 milliseconds apart
    # https://huggingface.co/kyutai/mimi, last visit 21.10.2026
    frame_duration_seconds = 1.0 / mimi.frame_rate
    # A model step is 80 milliseconds and now the question is how many samples are included in that time frame
    # This boils down to 24000/12.5 = 1920 audio samples per frame
    frame_size = int(sample_rate / mimi.frame_rate)
    device = lm_gen.lm_model.device

    # The audio sample rate was preprocessed to the needed 24 kHz.
    # Mimi expects only 1 audio channel per default therefore mono is being ensured.
    # https://huggingface.co/docs/transformers/model_doc/mimi
    input_wav, _ = audio_read(
        path_to_original_audio,
        mono=True,
    )

    # 48000 samples with a sample rate of 24000 equals 2 seconds of audio duration
    audio_sample_amount = input_wav.shape[-1]
    # Amount of frames that are needed for this audio.
    source_frame_count = (
                                 audio_sample_amount + frame_size - 1
                         ) // frame_size
    # Hibiki translates audio which has a maximum of 120sec * 12.5 = 1500 frames
    maximum_stream_frame_count = int(
        generation_window_seconds * mimi.frame_rate
    )

    generated_text_token_ids: list[int] = []

    output_audio_parts: list[torch.Tensor] = []

    # For output delay measurement
    # Begins counting when the input audio starts.
    output_audio_start_seconds: float | None = None
    # Begins counting when first output chunk is available.
    current_output_duration_seconds = 0.0
    eos_generated = False
    parts_after_last_text_token: int | None = None
    input_eos_sent = False

    silence_frame = torch.zeros(
        1,
        frame_size,
        dtype=input_wav.dtype,
    )

    # Activates a new streaming session without needing to reload the model.
    # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/modules/streaming.py#L132
    with lm_gen.streaming(batch_size=1), mimi.streaming(
            batch_size=1
    ):
        # Waits for GPU to finish before starting the new timer
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        input_stream_started_at = time.perf_counter()

        # first input audio frame is ignored in the official implementation
        # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L138-L144
        skip_frames = 1

        for step in range(maximum_stream_frame_count):

            # starting time + (0 + 1) * 0,08
            input_audio_deadline = (
                    input_stream_started_at
                    + (step + 1) * frame_duration_seconds
            )

            # (starting time + (0 + 1) * 0,08) - (starting time + time that passed)
            remaining_wait_seconds = (
                    input_audio_deadline - time.perf_counter()
            )

            # simulates the 80 milliseconds of input audio recording while performing live translation
            if remaining_wait_seconds > 0:
                time.sleep(remaining_wait_seconds)

            # if last frame of input audio frames was not reached
            if step < source_frame_count:
                # 0 = 0*1920
                frame_start = step * frame_size
                frame_end = min(
                    frame_start + frame_size,
                    # special case where last frame is reached and not enough audio samples are left to fill the frame
                    audio_sample_amount,
                )
                # loads the 80 millisecond frame (in case of the last frame it could be shorter)
                pcm_frame = input_wav[:, frame_start:frame_end]

                # calculation of the missing samples. If > 0 then audio is shorter than 1920 samples and needs to be padded with zeros
                missing_samples = frame_size - pcm_frame.shape[-1]
                if missing_samples > 0:
                    pcm_frame = torch.nn.functional.pad(
                        pcm_frame,
                        (0, missing_samples),
                    )
            else:
                pcm_frame = silence_frame

            # Adds a third dimension (batch dimension to the pcm_frame tensor)
            # When looking at the referenced code one can see that originally batch and channel was added this way.
            # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L136
            pcm_frame = pcm_frame.to(device=device)[None]

            # Encoding of input audio returns "... an int tensor of shape [B, K, T] ..." K represents the amount of audio tokens
            # of a single frame with each token representing a different audio detail.
            # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/compression.py#L383
            codes = mimi.encode(pcm_frame)

            if skip_frames:
                ### https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L139-142
                # "The first input audio frame is ignored, as from the point of
                # view of the model it is in the past. We still `mimi.encode` for simplicity,
                # however as the first encoded frame has a specific structure (due to the left padding),
                # we reset the streaming state of the encoder to reapply the padding on the next call."
                ###
                mimi.reset_streaming()
                skip_frames -= 1

            # Marks the end of the input audio and therefore adds an end of sequence token
            # In the original streaming code the closing is based on a closed web socket
            # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L105-L158 last visit: 02.11.2026
            # In the original translation of already available audio an end of sequence token is placed inside the fully available audio sequence
            # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L190 last visit: 02.11.2026
            # In this case there is no websocket to close and since I simulate a livestream frame by fram I also don't have the whole sequence
            # Therefore the whole "codes" variable of this frame is being set as an end of sequence token.
            # This is done by setting the codebook values to 2048 (out of the range of the normal codebook values)
            # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/quantization/vq.py#L144-145 last visit: 02.11.2026
            # place where cardinality 2048 is defined
            # https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L61
            if step == source_frame_count:
                codes = torch.full_like(
                    codes,
                    fill_value=mimi.cardinality,
                )
                input_eos_sent = True

            # codes.shape[-1] includes the information how many frames there are in "codes" and since each frame
            # is processed separately this value is typically 1. (I left the loop there because this process is also in the live implementation)
            # https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L145-148 last visit: 02.11.2026
            for code_index in range(codes.shape[-1]):

                #audio/text translation process
                #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py#L788 last visit: 04.11.2026
                #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py#L669 last visit: 04.11.2026
                tokens = lm_gen.step(
                    # Sends the first (and only) frame to lm_gen.step
                    codes[:, :, code_index:code_index + 1]
                )
                if tokens is None:
                    continue

                #Code as in:
                #https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L79-81 last visit:03.11.2026
                #The connection that dep_q is set to 16 is really hard to display therefore I only reference the source of 16 in the config file.
                #https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16/blob/main/config.json#L4 last visit: 04.11.2026
                #In general dep_q is the number of target audio codebook streams
                #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py#L54 last visit: 04.11.2026
                #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py#L188 last visit: 04.11.2026
                assert tokens.shape[1] == lm_gen.lm_model.dep_q + 1
                #"tokens" stores the 16 audio streams from index 1-16 (index 0 is text token)
                #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py#L762-L767 last visit: 05.11.2026
                #Therefore here only the audio codebook streams are being decoded
                decoded_chunk = mimi.decode(tokens[:, 1:]).cpu()[0]

                ########## Benchmark ##########

                #time that passed since the start of the input audio stream till the generation of the translated audio chunk
                chunk_ready_seconds = (
                        time.perf_counter()
                        - input_stream_started_at
                )

                #For the first frame of the audio sequence (important for TTFA measurement)
                if output_audio_start_seconds is None:
                    output_audio_start_seconds = chunk_ready_seconds
                else:
                    #To accurately calculate TTFA and End Offset (by force aligning time stamps to output words) it is necessary
                    #to add the processing time that exceeds the play time of the last output audio chunk as silence
                    #into the final output audio

                    #Time that passed since the first output chunk was ready and the moment the current output chunk was ready
                    chunk_available_in_output_seconds = (
                            chunk_ready_seconds
                            - output_audio_start_seconds
                    )

                    #Time that passed since the audio chunk that was calculated before stopped "playing".
                    #Chunk x ended at 1,24 seconds, Chunk x+1 was ready at 1,30 -> therefore a gap of 0,06 seconds needs
                    #to be filled with silence
                    gap_seconds = (
                            chunk_available_in_output_seconds
                            - current_output_duration_seconds #Sum of output audio durations and already occurred gaps
                    )

                    #Model runs on 24000 samples per second (see "sample_rate" declaration section)
                    gap_samples = max(
                        0,
                        round(gap_seconds * sample_rate),
                    )

                    if gap_samples > 0:
                        #Fills the output audio with silence
                        output_audio_parts.append(
                            torch.zeros(
                                decoded_chunk.shape[0],  #1 (Batch amount)
                                gap_samples,
                                dtype=decoded_chunk.dtype,
                                device=decoded_chunk.device,
                            )
                        )
                        #Adds the gaps to the output duration
                        current_output_duration_seconds += (
                                gap_samples / sample_rate
                        )

                #decoded output audio is added after the added silence gap
                output_audio_parts.append(decoded_chunk)
                current_output_duration_seconds += (
                        decoded_chunk.shape[-1] / sample_rate
                )
                ########## Benchmark end ##########

                #Right before the Benchmark code the audio streams where selected and decoded (line ~471) next step is to retrieve
                #the text token at index 0.
                #https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L85 last visit: 05.11.26
                text_token_id = int(tokens[0, 0, 0].item())
                #text_token_ids are collected
                generated_text_token_ids.append(text_token_id)

                #If EOS token is not being produced before the 120 * 12,5 frames ended, the model needs to know
                #when the last text token with actual content was produced
                #A token with content is one that has an ID > 3
                #Token with ID 3 is a Padding token
                #https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16/blob/main/config.json last visit: 7.11.26
                #As stated in row 164 hibiki uses the text tokenizer of Googles sentencepiece therefore:
                #Token ID 0 -> Unknown | ID 1 -> beginning of sequence | ID 2 -> end of sequence
                #https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md#customizing-default-special-symbols-unk-bos-eos-pad last visit: 7.11.26
                if text_token_id > text_tokenizer.pad_id():
                    parts_after_last_text_token = len(output_audio_parts)


                if (
                        input_eos_sent
                        and text_token_id == text_tokenizer.eos_id() #As stated above text_tokenizer.eos_id() == 2
                ):
                    eos_generated = True
                    break

            if eos_generated:
                break

    if not output_audio_parts or output_audio_start_seconds is None:
        raise RuntimeError(
            f"Hibiki produced no streaming audio for "
            f"{path_to_original_audio}."
        )

    #Cuts of all text_token_ids that are after the EOS token
    if text_tokenizer.eos_id() in generated_text_token_ids:
        output_eos_index = generated_text_token_ids.index(
            text_tokenizer.eos_id()
        )
        text_tokens_to_decode = generated_text_token_ids[:output_eos_index]
    else:
        warnings.warn(
            f"Hibiki produced no output EOS token for "
            f"{path_to_original_audio} within "
            f"{generation_window_seconds:.1f} seconds. The streaming output "
            f"is truncated after the last generated text token.",
            RuntimeWarning,
        )
        if parts_after_last_text_token is None:
            raise RuntimeError(
                f"Hibiki produced neither an EOS token nor a non-padding "
                f"text token for {path_to_original_audio}."
            )
        #Uses the value calculated in row 547 to cut of audio parts that are after the last text token that included translated text.
        output_audio_parts = output_audio_parts[
            :parts_after_last_text_token
        ]
        text_tokens_to_decode = generated_text_token_ids

    #Filters text tokens that don't hold translated text content
    #https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L250 last visit: 08.11.2026
    translated_text_token_ids = [
        token_id
        for token_id in text_tokens_to_decode
        if token_id > text_tokenizer.pad_id()
    ]

    #From ids to actual words
    #https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L251
    translated_text = text_tokenizer.decode(
        translated_text_token_ids
    )

    #Final output audio which includes the output chunks as well as the added silent audio gaps
    output_wav = torch.cat(
        output_audio_parts,
        dim=-1,
    )

    ########## Preprocessing the sample rate to 16kHz for the Whisper model ##########
    import subprocess

    whisper_sample_rate = 16_000
    #(240000 / 24000) * 16000 = 160000 samples
    expected_whisper_sample_count = round(
        output_wav.shape[-1] * whisper_sample_rate / sample_rate
    )

    output_audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ffmpeg_filter = (
        #Resamples the input (output audio of hibiki) to 16kHz
        #https://ffmpeg.org/ffmpeg-filters.html#aresample last visit: 08.11.2026
        f"aresample={whisper_sample_rate},"
        #Adds silence to the end if the output has less than the expected sample count
        #https://ffmpeg.org/ffmpeg-filters.html#apad last visit: 08.11.2026
        f"apad=whole_len={expected_whisper_sample_count}," 
        #Cuts of samples that are over the expected sample count
        #https://ffmpeg.org/ffmpeg-filters.html#atrim last visit: 08.11.2026
        f"atrim=end_sample={expected_whisper_sample_count}"
    )
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",                    #"Overwrite output files without asking." https://ffmpeg.org/ffmpeg.html#Main-options last visit: 08.11.2026
        "-f", "f32le",           #Defines the input format as float 32 bit little endian == "<f4" (See below at subprocess.run) https://ffmpeg.org/ffmpeg.html#Main-options last visit: 08.11.2026
        "-ar", str(sample_rate), #Defines the input sample rate as 24kHz https://ffmpeg.org/ffmpeg.html#Audio-Options last visit: 08.11.2026
        "-ac", "1",              #Defines the input audio as being mono https://ffmpeg.org/ffmpeg.html#Audio-Options last visit: 08.11.2026
        "-i", "pipe:0",
        "-af", ffmpeg_filter,    #Selects ffmpeg_filter as filtergraph https://ffmpeg.org/ffmpeg.html#Audio-Options last visit: 08.11.2026
        "-c:a", "pcm_s16le",
        str(output_audio_path),
    ]

    try:
        subprocess.run(
            ffmpeg_command,
            #<f4 == little-endian 32-bit floating-point number
            #https://numpy.org/doc/stable/reference/arrays.dtypes.html#index-6 last visit: 08.11.2026
            input=output_wav.numpy().astype("<f4", copy=False).tobytes(),
            check=True,
        )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"FFmpeg failed to create the 16 kHz Whisper WAV at "
            f"{output_audio_path}."
        ) from error

    return GeneratedTranslation(
        translated_text=translated_text,
        path_to_translated_audio=output_audio_path,
        output_audio_start_delay=output_audio_start_seconds,
    )


if __name__ == "__main__":
    main()
