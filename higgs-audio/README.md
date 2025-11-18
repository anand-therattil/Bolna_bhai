<h1 align="center">Higgs Audio V2: Redefining Expressiveness in Audio Generation</h1>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://boson.ai/blog/higgs-audio-v2"><img src='https://img.shields.io/badge/🚀-Launch Blogpost-228B22' style="margin-right: 5px;"></a>
  <a href="https://boson.ai/demo/tts"><img src="https://img.shields.io/badge/🕹️-Boson%20AI%20Playground-9C276A" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/spaces/smola/higgs_audio_v2"><img src="https://img.shields.io/badge/🎮-HF%20Space%20Playground-8A2BE2" style="margin-right: 5px;"></a>
  <a href="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base"><img src="https://img.shields.io/badge/🤗-Checkpoints (3.6B LLM + 2.2B audio adapter)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

# Training repo for Higgs Audio v2  

## 🎥 Tutorial Videos

| Platform | Link |
|----------|------|
| YouTube | [![Tutorial](http://img.youtube.com/vi/u7og6yAx91g/0.jpg)](https://www.youtube.com/watch?v=u7og6yAx91g) |
| 哔哩哔哩 | [![训练脚本教程](https://i0.hdslb.com/bfs/archive/placeholder.jpg)](https://www.bilibili.com/video/BV1zaYnzoEoD/) |

> 📖 Screen recording videos showing how to use the training and generation scripts for Higgs Audio v2.



# Data Processing and Training Guide  
数据处理与训练指南  

⚠️ Note: Currently, only single-speaker training is implemented  

## NEW  

- New language training  
- Experimental feature: DDP support. For details, please refer to: `DDP_training.sh`  
- Optimized input parameters, removed unnecessary misleading parameters  
- Adopted official data classes  
- Supports LoRA training, 16G memory is sufficient for training  
- Provides a mini training set, welcome to use  

## TODO  
- [ ] Multi-speaker training

## Training Environment Setup 训练环境配置

### Option 3: Using conda
```bash
git clone https://github.com/JimmyMa99/train-higgs-audio.git
cd train-higgs-audio

conda create -n higgs_audio_env python=3.10
conda activate higgs_audio_env
pip install -r requirements_train.txt
pip install -e .
```

## Data Processing  数据处理

First, prepare your audio and text data in the required format.  
首先，请按照要求准备好音频和文本数据。

### Data Format  数据格式

ms-swift data format:  
ms-swift 数据格式:
```jsonl
{"messages": [{"role": "assistant", "content": "<audio>描述了今天天气真不错"}], "audios": ["/xxx/x.wav"]}
```

Run the script  
运行脚本

```shell
python tools/convert_jsonl_to_higgs.py \
  --jsonl_files /path/to/audio.jsonl \
  --output_dir ./higgs_training_data \
  --copy_audio True
```

Obtain data in the following format  
得到以下格式的数据

```shell
higgs_training_data/
├── metadata.json                  # Overall metadata file of the dataset
├── huo_speaker_000001.wav         # Audio file 1 of speaker "huo"
├── huo_speaker_000001.txt         # Text transcription corresponding to the audio
├── huo_speaker_000002.wav         # Audio file 2 of speaker "huo"
├── huo_speaker_000002.txt         # Text transcription corresponding to the audio
├── ...                            # More audio/text files of "huo_speaker"
├── huo_speaker_000051.wav         # Audio file 1 of speaker "huo"
├── huo_speaker_000051.txt         # Text transcription corresponding to the audio
├── huo_speaker_000052.wav         # Audio file 2 of speaker "huo"
├── huo_speaker_000052.txt         # Text transcription corresponding to the audio
└── ...                            # More audio/text files of "huo_speaker"
```

metadata.json 格式
```json
{
  "dataset_info": {
    "total_samples": 2797,
    "speakers": [
      "huo_speaker"
    ],
    "languages": [
      "zh"
    ],
    "total_duration": 12173.9,
    "avg_duration": 4.35,
    "created_from": [
      "/root/code/new_work_code/HI-TransPA/swfit_workdir/fresh-little-lemon-workspace/data/swift_format/huo_audio.jsonl"
    ]
  },
  "samples": [
    {
      "id": "huo_speaker_000000",
      "audio_file": "huo_speaker_000000.wav",
      "transcript_file": "huo_speaker_000000.txt",
      "duration": 3.86,
      "speaker_id": "huo_speaker",
      "speaker_name": "Huo",
      "scene": "recording_system",
      "emotion": "alerting",
      "ref_audio_file": If you need a reference tone color, please add this field, which will take effect under the "zero_shot_voice_cloning" model. 如果你是需要有参考音色，请加入此字段，这会在"zero_shot_voice_cloning"模型下生效
      "language": "zh",
      "gender": "unknown",
      "quality_score": 1.0,
      "original_audio_path": "audio_splits_huo/14_cropped_with_audio_line000001_vid00_f7b81293.wav",
      "user_instruction": "<audio> /translate",
      "task_type": "audio_generation"
    },
    {
      "id": "huo_speaker_000001",
      "audio_file": "huo_speaker_000001.wav",
      "transcript_file": "huo_speaker_000001.txt",
      "duration": 3.2,
      "speaker_id": "huo_speaker",
      "speaker_name": "Huo",
      "scene": "quiet_room",
      "emotion": "questioning",
      "ref_audio_file": If you need a reference tone color, please add this field, which will take effect under the "zero_shot_voice_cloning" model. 如果你是需要有参考音色，请加入此字段，这会在"zero_shot_voice_cloning"模型下生效
      "language": "zh",
      "gender": "unknown",
      "quality_score": 1.0,
      "original_audio_path": "audio_splits_huo/126_cropped_with_audio_line000002_vid00_66220ae5.wav",
      "user_instruction": "<audio> /translate",
      "task_type": "audio_generation"
    }
  ]
}

```


## Training  训练

Please make sure to modify all parameters before training, including data path, model path, number of training epochs, etc.  
请务必在训练前修改各个参数，包括数据路径、模型路径、训练轮数等。

```shell
python trainer/trainer.py
```

Fine-tuning with LoRA requires the use of `--use_lora`, like:

```shell
python trainer/trainer.py --use_lora
```

It should be noted that when using LoRA to fine-tune new voices, there may be cases where normal output cannot be achieved. This issue has currently been found in the migration fine-tuning of Vietnamese, and it is not yet clear whether it is a training problem or other circumstances. Based on past experience, when training a model to learn knowledge it has never been exposed to, it is better to use full fine-tuning with the parameter `--use_lora False`.





## Merge lora
```shell
bash merge_model.sh \
    --base_model_path xxx \
    --lora_adapter_path xxx \
    --output_path xxx \
    --compare_models \
    --test_input "A custom sentence for testing." 
```

## generate  生成

```shell
bash generate.sh
```

## Experiment Comparison: Text and Audio Effect Comparison  实验对比：文本与音频效果对照

To intuitively show the difference between generated sounds and real sounds, the following table contains directly playable audio files:  
为直观展示生成声音与真实声音的差异，以下表格包含可直接播放的音频文件：

Since the data I have is the speech of hearing-impaired individuals, for the purpose of comparison, I selected a speech sample from a hearing-impaired person as the real voice, and a generated version of the same speech as the generated voice.
因为我手上的数据是听障人士的语音，因此在对比时，我选择了一个听障人士的语音作为真实声音，另一个相同语音的生成版本作为生成声音。



| text 文本内容 | real record 真实声音（用户后录） | generate record生成声音（脚本输出） |
|----------|----------------------|----------------------|
| 大家好，我是火君，我居住在上海 | [点击播放/下载 (huojun.MP3)](test_demo/huojun.MP3) | [点击播放/下载 (huojun_gen.wav)](test_demo/huojun_gen.wav) |
| 我爱机智流，机智流是最好的开源社区 | [点击播放/下载 (smartflowai.MP3)](test_demo/smartflowai.MP3) | [点击播放/下载 (smartflowai_gen.wav)](test_demo/smartflowai_gen.wav) |
| tôi cũng như là những người lính như | [点击播放/下载 (vn_demo.MP3)](test_demo/vn_demo.MP3) | [点击播放/下载 (vn_gen.wav)](test_demo/vn_gen.wav) |

训练前后对比(没有使用参考音频)
| text 文本内容 | before training 训练前 | after training 训练后 |
|----------|----------------------|----------------------|
| 你好，我是火君 | [点击播放/下载 (huojun.MP3)](test_demo/generation_for_huo_no_ref_no_train.wav) | [点击播放/下载 (huojun_gen.wav)](test_demo/generation_for_huo_no_ref.wav) |

We are open-sourcing Higgs Audio v2, a powerful audio foundation model pretrained on over 10 million hours of audio data and a diverse set of text data. Despite having no post-training or fine-huoing, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

On [EmergentTTS-Eval](https://github.com/boson-ai/emergenttts-eval-public), it achieves win rates of **75.7%** and **55.7%** over "gpt-4o-mini-tts" on the "Emotions" and "Questions" categories, respectively. It also obtains state-of-the-art performance on traditional TTS benchmarks like Seed-TTS Eval and Emotional Speech Dataset (ESD). Moreover, the model demonstrates capabilities rarely seen in previous systems, including generating natural multi-speaker dialogues in multiple languages, automatic prosody adaptation during narration, melodic humming with the cloned voice, and simultaneous generation of speech and background music.

<p align="center">
    <img src="figures/emergent-tts-emotions-win-rate.png" width=900>
</p>

Here's the demo video that shows some of its emergent capabilities (remember to unmute):

<video src="https://github.com/user-attachments/assets/0fd73fad-097f-48a9-9f3f-bc2a63b3818d" type="video/mp4" width="80%" controls>
</video>

Here's another demo video that show-cases the model's multilingual capability and how it enabled live translation (remember to unmute):

<video src="https://github.com/user-attachments/assets/2b9b01ff-67fc-4bd9-9714-7c7df09e38d6" type="video/mp4" width="80%" controls>
</video>

## Installation

We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. Following are two docker images that we have verified:
- nvcr.io/nvidia/pytorch:25.02-py3
- nvcr.io/nvidia/pytorch:25.01-py3

Here's an example command for launching a docker container environment. Please also check the [official NVIDIA documentations](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

```bash
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash
```

### Option 1: Direct installation


```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

pip install -r requirements.txt
pip install -e .
```

### Option 2: Using venv

```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

python3 -m venv higgs_audio_env
source higgs_audio_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


### Option 3: Using conda
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

conda create -n higgs_audio_env python=3.10
conda activate higgs_audio_env
pip install -r requirements.txt
pip install -e .
```

### Option 4: Using uv
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### Option 5: Using vllm

For advanced usage with higher throughput, we also built OpenAI compatible API server backed by vLLM engine for you to use.
Please refer to [examples/vllm](./examples/vllm) for more details.


## Usage

> [!TIP]
> For optimal performance, run the generation examples on a machine equipped with GPU with at least 24GB memory!

### Get Started

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use Higgs Audio v2.

### Zero-Shot Voice Cloning
Generate audio that sounds similar as the provided [reference audio](./examples/voice_prompts/belinda.wav).

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--out_path generation.wav
```

The generation script will automatically use `cuda:0` if it founds cuda is available. To change the device id, specify `--device_id`:

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio belinda \
--temperature 0.3 \
--device_id 0 \
--out_path generation.wav
```

You can also try other voices. Check more example voices in [examples/voice_prompts](./examples/voice_prompts). You can also add your own voice to the folder.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--ref_audio broom_salesman \
--temperature 0.3 \
--out_path generation.wav
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years." \
--temperature 0.3 \
--out_path generation.wav
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```


## Technical Details
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finehuoed on top of [Higgs Audio v1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of Higgs Audio v2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt Higgs Audio v2 with the reference text, reference audio, and target text for zero-shot TTS. We use the standard evaluation metrics from Seed-TTS Eval and ESD.

|                              | SeedTTS-Eval| | ESD   |                 |
|------------------------------|--------|--------|---------|-------------------|
|                              | WER ↓ | SIM ↑ | WER ↓ | SIM (emo2vec) ↑ |
| Cosyvoice2                   | 2.28   | 65.49  | 2.71    | 80.48             |
| Qwen2.5-omni†                | 2.33   | 64.10  | -       | -                 |
| ElevenLabs Multilingual V2   | **1.43**   | 50.00  | 1.66    | 65.87             |
| Higgs Audio v1                | 2.18   | 66.27  | **1.49**    | 82.84             |
| Higgs Audio v2 (base)         | 2.44   | **67.70**  | 1.78    | **86.13**         |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice. The judge model is Gemini 2.5 Pro.

| Model                              | Emotions (%) ↑ | Questions (%) ↑ |
|------------------------------------|--------------|----------------|
| Higgs Audio v2 (base)               | **75.71%**   | **55.71%**         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)       | 61.64%       | 47.85%         |
| [Hume.AI](https://www.hume.ai/research)                            | 61.60%       | 43.21%         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)  | 50.00%       | 50.00%         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)      | 41.60%       | 51.78%         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)               | 40.86%        | 47.32%         |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)         | 30.35%       | 39.46%         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)                    | 29.28%       | 48.21%         |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                      | 15.96%       | 31.78%         |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of Higgs Audio v2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. We fix two reference audio clips to evaluate the model's ability in double voice cloning for utterances ranging from 4 to 10 dialogues between two randomly chosen persona.
- `small talk (no ref)`: 250 synthetic dialogues curated in the same way as above, but are characterized by short utterances and a limited number of turns (4–6), we do not fix reference audios in this case and this set is designed to evaluate the model's ability to automatically assign appropriate voices to speakers.
- `small talk (ref)`: 250 synthetic dialogues similar to above, but contains even shorter utterances as this set is meant to include reference clips in it's context, similar to `two-speaker-conversation`.


We report the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than Higgs Audio v2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626), two of the most popular open-source models capable of multi-speaker dialog generation. Results are summarized in the following table. We are not able to run [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances and output audio.

|                                                | two-speaker-conversation |                |small talk |                | small talk (no ref) |                |
| ---------------------------------------------- | -------------- | ------------------ | ---------- | -------------- | ------------------- | -------------- |
|                                                | WER ↓                      | Mean Sim & Dis-sim ↑ | WER ↓       |  Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast) | 38.77                    | 46.02         | **8.33**       | 63.68          | 24.65               | 53.94 |
| [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)         | \-                       | \-             | 17.62      | 63.15          | 19.46               | **61.14**          |
| Higgs Audio v2 (base)     | **18.88**                    | **51.95**          | 11.89      | **67.92**              | **14.65**               | 55.28              |


## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.
