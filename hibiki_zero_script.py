import io
import csv
from dataclasses import dataclass
from pathlib import Path

import sentencepiece
import torch
from hibiki_zero.run import DEFAULT_REPO

from moshi.models import LMGen, MimiModel, loaders
from hibiki_zero.inference import get_lmgen, seed_all

#https://github.com/kyutai-labs/hibiki-zero/blob/871e89d078202c7d9d18d0924bd76cf161cd6606/hibiki_zero/inference.py#L35
@dataclass
class LoadedModel:
    mimi_instance: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start_seconds: float
    end_seconds: float

@dataclass(frozen=True)
class TTFAAndEndOffsetMeasurements:
    ttfa: float
    end_offset: float


@dataclass(frozen=True)
class GeneratedTranslation:
    translated_text: str
    path_to_translated_audio: Path
    output_audio_start_delay: float
    # Time that passed since the start of the input audio till the first chunk of the output audio (maybe silence)

@dataclass
class ModelResult:
    path_to_original_audio: Path
    expected_translation_of_original_audio: str
    translated_text: str
    path_to_translated_audio: str

    # Time that passed from the beginning of the first input word till the beginning of the first output word
    ttfa: float| None
    # Time that passed from the end of the last input word till the end of the last output word
    end_offset: float| None

    bleu: float| None = None
    asr_bleu: float| None = None
    asr_comet: float| None = None

model_results: list[ModelResult] = []

def main() -> None:
    dataset_audio_path_transcription_and_translation_data = loading_dataset()
    model = loading_model()
    results = run_model(dataset_audio_path_transcription_and_translation_data, model)
    analysation_of_output_accuracy(results)


#Step one: Making the Dataset Accessible:
def loading_dataset() -> list[dict[str, str]]:
    with io.open("../Bridge_The_Knowledge_Gap.tsv", "r", encoding="utf-8", newline="") as dataset_tsv_file:
        #DictReader leaves out the fieldnames (1st row) https://docs.python.org/3/library/csv.html#csv.DictReader
        return list(csv.DictReader(dataset_tsv_file, delimiter="\t"))


#Step two: Loading the model (Build this method by analyzing the hibiki-zero gitlab repo (see links below)).
def loading_model(device: str = "cuda", fuse_lora=True, ) -> LoadedModel:


    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L168 last visit: 12.10.2026
    seed_all(42)

    hf_repo_parts = DEFAULT_REPO.split("@")
    hf_repo_name = hf_repo_parts[0]
    revision = hf_repo_parts[1]

    #Contains paths to checkpoint data (LLM-Weights)/sub models
    #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L148 last visit: 12.10.2026
    #---
    #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L181 last visit: 12.10.2026
    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L236 last visit: 12.10.2026
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=hf_repo_name,
        revision=revision,
    )

    #Returns the mimi model and Loads the model weights onto the cuda device.
    #https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py#L282 last visit: 12.10.2026
    #---
    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L247 last visit: 12.10.2026
    mimi = checkpoint_info.get_mimi(device=device)

    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L248 last visit: 12.10.2026
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L251 last visit: 12.10.2026
    lm = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16, fuse_lora=fuse_lora)

    #https://github.com/kyutai-labs/hibiki-zero/blob/main/hibiki_zero/run.py#L267 last visit: 12.10.2026
    lm_gen: LMGen = get_lmgen(lm=lm, checkpoint_info=checkpoint_info, batch_size=1)

    return LoadedModel(
        mimi_instance=mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
    )

#Step three: Translating Audio Files into Text and Audio while Measuring Real-Time Capabilities
def run_model(dataset: list[dict[str, str]],
              loaded_model: LoadedModel,
              ) -> list[ModelResult] :

    translated_audio_output_directory: Path = Path("translated_audio")

    translated_audio_output_directory.mkdir(parents=True, exist_ok=True)

    for row_number, row_data in enumerate(
        dataset,
        start=1
    ):
        path_to_original_audio = Path(row_data["path"])
        original_transcription = row_data["sentence"]
        expected_translation_of_original_audio = row_data["translation"]


        output_audio_path = (
                translated_audio_output_directory
                / f"{row_number:06d}_{path_to_original_audio.stem}.wav"
        )

        #Hibiki translation
        generated_translation = generate_translation(
            path_to_original_audio=path_to_original_audio,
            output_audio_path=output_audio_path,
            loaded_model=loaded_model,
        )

        #Timestamping words of original audio
        original_word_timestamps = word_timestamping(
            path_to_original_audio=path_to_original_audio,
            original_transcription=original_transcription
        )

        #Timestamping words of output audio
        output_word_timestamps = word_timestamping(
            generated_translation.path_to_translated_audio,
            generated_translation.translated_text,
        )

        #Measuring latency between the audios (beware that generated_translation holds info about when the output audio
        #started after the original audio started)
        measurements = measure_ttfa_and_end_offset(
            original_word_timestamps=original_word_timestamps,
            output_word_timestamps=output_word_timestamps,
            output_audio_start_delay=generated_translation.output_audio_start_delay,
        )

        #Adding the ttfa and end offset measurements
        model_results.append(
            ModelResult(
                path_to_original_audio=path_to_original_audio,
                expected_translation_of_original_audio=expected_translation_of_original_audio,
                translated_text= generated_translation.translated_text,
                path_to_translated_audio=str(generated_translation.path_to_translated_audio),
                ttfa=measurements.ttfa,
                end_offset=measurements.end_offset,
            )
        )

    return model_results

def generate_translation(path_to_original_audio: Path,
                         output_audio_path: Path,
                         loaded_model: LoadedModel
                        ) -> GeneratedTranslation:

    return GeneratedTranslation()


def word_timestamping(path_to_original_audio: Path,
                      original_transcription: str) -> WordTimestamp:

    return WordTimestamp()

def measure_ttfa_and_end_offset(original_word_timestamps: WordTimestamp,
                                output_word_timestamps: WordTimestamp,
                                output_audio_start_delay: float
                                ) -> TTFAAndEndOffsetMeasurements:
    return TTFAAndEndOffsetMeasurements()


#Step four: Measuring Output Accuracy
def analysation_of_output_accuracy(
        model_results: list[ModelResult]
) -> None:

    for model_result in model_results:

        translated_text = model_result.translated_text
        path_to_translated_audio = model_result.path_to_translated_audio

        #BLEU

        #ASR BLEU

        #ASR COMET


if __name__ == "__main__":
    main()