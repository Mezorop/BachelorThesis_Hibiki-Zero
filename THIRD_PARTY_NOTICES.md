# Third-party notices 
[written by codex - reviewed by repository author]

## Hibiki-Zero source code

Parts of [`hibiki_zero_script.py`](hibiki_zero_script.py), in particular the
model-loading and inference/generation logic, are adapted from Hibiki-Zero:

- Project: Hibiki-Zero
- Copyright: Copyright (c) 2026 kyutai
- Source: <https://github.com/kyutai-labs/hibiki-zero>
- Source revision used for the adaptation:
  `871e89d078202c7d9d18d0924bd76cf161cd6606`
- License: MIT License

The adapted portions have been modified for repeated dataset evaluation,
real-time timing measurements, and reconstruction of the generated audio
timeline. The complete MIT license text is reproduced in
[`LICENSES/Hibiki-Zero-MIT.txt`](LICENSES/Hibiki-Zero-MIT.txt).

## Hibiki-Zero model weights

At runtime, Hibiki-Zero obtains the model identified by `DEFAULT_REPO`, which
currently refers to Kyutai's `hibiki-zero-3b-pytorch-bf16` model. The weights
are not included in this repository and have a license separate from the
source code:

- Model: Hibiki-Zero 3B
- Developed by: Kyutai
- Model page: <https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16>
- Model revision used:
  `23b3e0b41782026c81dd5283a034107b01f9e513`
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)
- License text: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>

Use, redistribution, and adaptation of the model weights must comply with
that model license, including its attribution, non-commercial-use, and
share-alike conditions. The MIT license for the Hibiki-Zero source code does
not apply to the model weights.

The notices above cover the identified third-party material only. They do not
declare a license for the repository's independently written code.
