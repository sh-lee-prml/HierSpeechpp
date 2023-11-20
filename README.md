# HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation by Hierarchical Variational Inference for Zero-shot Speech Synthesis
The official implementation of HierSpeech2 | [Paper]() | [Demo page](https://sh-lee-prml.github.io/HierSpeechpp-demo/) | [Checkpoint](https://drive.google.com/drive/folders/1-L_90BlCkbPyKWWHTUjt5Fsu3kz0du0w?usp=sharing) | [<svg xmlns="http://www.w3.org/2000/svg" width="95" height="88" fill="none">
	<path
		fill="#fff"
		d="M94.25 70.08a8.28 8.28 0 0 1-.43 6.46 10.57 10.57 0 0 1-3 3.6 25.18 25.18 0 0 1-5.7 3.2 65.74 65.74 0 0 1-7.56 2.65 46.67 46.67 0 0 1-11.42 1.68c-5.42.05-10.09-1.23-13.4-4.5a40.4 40.4 0 0 1-10.14.03c-3.34 3.25-7.99 4.52-13.39 4.47a46.82 46.82 0 0 1-11.43-1.68 66.37 66.37 0 0 1-7.55-2.65c-2.28-.98-4.17-2-5.68-3.2a10.5 10.5 0 0 1-3.02-3.6c-.99-2-1.18-4.3-.42-6.46a8.54 8.54 0 0 1-.33-5.63c.25-.95.66-1.83 1.18-2.61a8.67 8.67 0 0 1 2.1-8.47 8.23 8.23 0 0 1 2.82-2.07 41.75 41.75 0 1 1 81.3-.12 8.27 8.27 0 0 1 3.11 2.19 8.7 8.7 0 0 1 2.1 8.47c.52.78.93 1.66 1.18 2.61a8.61 8.61 0 0 1-.32 5.63Z"
	/>
	<path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z" />
	<path
		fill="#FF9D0B"
		d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"
	/>
	<path
		fill="#3A3B45"
		d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"
	/>
	<path
		fill="#FF323D"
		d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"
	/>
	<path
		fill="#3A3B45"
		fill-rule="evenodd"
		d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z"
		clip-rule="evenodd"
	/>
	<path
		fill="#FF9D0B"
		d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"
	/>
	<path
		fill="#FFD21E"
		d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"
	/>
	<path
		fill="#FF9D0B"
		d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"
	/>
	<path
		fill="#FFD21E"
		d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"
	/>
</svg>
![huggingface_logo](https://github.com/sh-lee-prml/HierSpeechpp/assets/56749640/a9629a89-3d3c-40b3-a438-7e49dff9c8ff)
HuggingFace](https://huggingface.co/spaces/HierSpeech/HierSpeech_TTS)

**Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee**

 Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract
![image](https://github.com/sh-lee-prml/HierSpeechpp/assets/56749640/732bc183-bf11-4f32-84a9-e9eab8190e1a)
<details> 
<summary> [Abs.] Sorry for too long abstract😅 </summary>


Recently, large-scale language models (LLM)-based speech synthesis has shown a significant performance in zero-shot speech synthesis. However, they require a large-scale data and even suffer from the same limitation of previous autoregressive speech models such as slow inference speed and lack of robustness. Following the previous powerful end-to-end text-to-speech framework of VITS (but now that's what we call classical), this paper proposes HierSpeech++, a fast and strong zero-shot speech synthesizer for text-to-speech (TTS) and voice conversion (VC). In the previous our works (HierSpeech and HierVST), we verified that hierarchical speech synthesis frameworks could significantly improve the robustness and expressiveness of the synthetic speech by adopting hierarchical variational autoencoder and leveraging self-supervised speech represenation as an additional linguistic information to bridge an information gap between text and speech. In this work, we once again significantly improve the naturalness and speaker similarity of the synthetic speech even in the zero-shot speech synthesis scenarios. We first introduce multi-audio acoustic encoder for the enhanced acoustics posterior, and adopt a hierarchical adaptive waveform generator with conditional/unconditional generation. Second, we additionally utilize a F0 information and introduce source-filter theory-based multi-path semantic encoder for speaker-agnostic and speaker-related semantic representation. We also leverage hierarchical variational autoencoder to connect multiple representations, and present a BiT-Flow which is a bidirectional normalizing flow Transformer networks with AdaLN-Zero for better speaker adaptation and train-inference mismatch reduction. Without any text transcripts, we only utilize the speech dataset to train the speech synthesizer for data flexibility. For text-to-speech, we introduce text-to-vec (TTV) frameworks to generate a self-supervised speech representation and F0 representation from text representation and prosody prompt. Then, the speech synthesizer of HierSpeech++ generates the speech from generated vector, F0, and voice prompt. In addition, we propose the high-efficient speech super-resolution framework which can upsample the waveform audio from 16 kHz to 48 kHz, and this facilitate training the speech synthesizer in that we can use easily available low-resolution (16 kHz) speech data for scaling-up. The experimental results demonstrated that hierarchical variational autoencoder could be a strong zero-shot speech synthesizer by beating LLM-based models and diffusion-based models for TTS and VC tasks. Furthermore, we also verify the data efficiency in that our model trained with a small dataset still shows a better performance in both naturalness and similarity than other models trained with large-scale dataset. Moreover, we achieve the first human-level quality in zero-shot speech synthesis.
</details>

This repository contains:

- 🪐 A PyTorch implementation of HierSpeech++ (TTV, Hierarchical Speech Synthesizer, SpeechSR)
- ⚡️ Pre-trained HierSpeech++ models trained on LibriTTS (Train-460, Train-960, and more dataset)

<!--
- 💥 A Colab notebook for running pre-trained HierSpeech++ models (Soon..)
🛸 A HierSpeech++ training script (Will be released soon)
-->
## Previous Our Works
- [1] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis
- [2] HierVST: Hierarchical Adaptive Zero-shot Voice Style Transfer

This paper is an extenstion version of above papers.

## Todo
### Hierarchical Speech Synthesizer
- [x] HierSpeechpp-Backbone
<!--
- [ ] HierSpeech-Lite (Fast and Efficient Zero-shot Speech Synthesizer)
- [ ] HierSinger (Zero-shot Singing Voice Synthesizer)
- [ ] HierSpeech2-24k-Large-Full (For High-resolutional and High-quality Speech Synthesizer)
- [ ] HierSpeech2-48k-Large-Full (For Industrial-level High-resolution and High-quality Speech Synthesizer)
-->
### Text-to-Vec (TTV)
- [x] TTV-v1 (LibriTTS-train-960)
- [ ] TTV-v2 (We are currently training a multi-lingual TTV model)  
<!--
- [ ] Hierarchical Text-to-Vec (For Much More Expressive Text-to-Speech)
-->
### Speech Super-resolution (16k --> 24k or 48k) 
- [x] SpeechSR-24k
- [x] SpeechSR-48k
### Training code (Will be released after paper acceptance)
- [ ] TTV
- [ ] Hierarchical Speech Synthesizer
- [ ] SpeechSR
## Getting Started

### Pre-requisites
0. Pytorch >=1.13 and torchaudio >= 0.13
1. Install requirements
```
pip install -r requirements.txt
```
2. Install Phonemizer
```
pip install phonemizer
sudo apt-get install espeak-ng
```
   
## Checkpoint [[Download]](https://drive.google.com/drive/folders/1-L_90BlCkbPyKWWHTUjt5Fsu3kz0du0w?usp=sharing)
### Hierarchical Speech Synthesizer
| Model |Sampling Rate|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HierSpeech2|16 kHz|97M| LibriTTS (train-460) |245|1,151|[[Download]](https://drive.google.com/drive/folders/14FTu0ZWux0zAD7ev4O1l6lKslQcdmebL?usp=sharing)|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960) |555|2,311|[[Download]](https://drive.google.com/drive/folders/1sFQP-8iS8z9ofCkE7szXNM_JEy4nKg41?usp=drive_link)|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960), Libri-light (Small, Medium), Expresso, MMS(Kor), NIKL(Kor)|2,796| 7,299 |[[Download]](https://drive.google.com/drive/folders/14jaDUBgrjVA7bCODJqAEirDwRlvJe272?usp=drive_link)|

<!--
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960))  |-|
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960) NIKL, AudioBook-Korean)  |-|
| HierSpeech2-Large-CL|16 kHz|200M| LibriTTS (train-960), Libri-Light, NIKL, AudioBook-Korean, Japanese, Chinese, CSS, MLS)  |-|
-->

### TTV
| Model |Language|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| TTV |Eng|107M| LibriTTS (train-960) |555|2,311|[[Download]](https://drive.google.com/drive/folders/1QiFFdPhqhiLFo8VXc0x7cFHKXArx7Xza?usp=drive_link)|


<!--
| TTV |Kor|100M| NIKL |114|118|-|
| TTV |Eng|50M| LibriTTS (train-960) |555|2,311|-|
| TTV-Large |Eng|100M| LibriTTS (train-960) |555|2,311|-|
| TTV-Lite |Eng|10M| LibriTTS (train-960) |555|2,311|-|
| TTV |Kor|50M| NIKL |114|118|-|
-->
### SpeechSR
| Model |Sampling Rate|Params|Dataset |Checkpoint|
|------|:---:|:---:|:---:|:---:|
| SpeechSR-24k |16kHz --> 24 kHz|0.13M| LibriTTS (train-960), MMS (Kor) |[speechsr24k](https://github.com/sh-lee-prml/HierSpeechpp/blob/main/speechsr24k/G_340000.pth)|
| SpeechSR-48k |16kHz --> 48 kHz|0.13M| MMS (Kor), Expresso (Eng), VCTK (Eng)|[speechsr48k](https://github.com/sh-lee-prml/HierSpeechpp/blob/main/speechsr48k/G_100000.pth)|

## Text-to-Speech
```
sh inference.sh

# --ckpt "logs/hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth" \ LibriTTS-460
# --ckpt "logs/hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth" \ LibriTTS-960
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth" \ Large_v1 epoch 60 (paper version)
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \ Large_v1.1 epoch 200 (20. Nov. 2023)

CUDA_VISIBLE_DEVICES=0 python3 inference.py \
                --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \
                --ckpt_text2w2v "logs/ttv_libritts_v1/ttv_lt960_ckpt.pth" \
                --output_dir "tts_results_eng_kor_v2" \
                --noise_scale_vc "0.333" \
                --noise_scale_ttv "0.333" \
                --denoise_ratio "0"

```
- For better robustness, we recommend a noise_scale of 0.333
- For better expressiveness, we recommend a noise_scale of 0.667
- Find your best parameters for your style prompt 😵
### Noise Control 
```
# without denoiser
--denoise_ratio "0"
# with denoiser
--denoise_ratio "1"
# Mixup (Recommended 0.6~0.8)
--denoise_rate "0.8" 
```
## Voice Conversion
- This method only utilize a hierarchical speech synthesizer for voice conversion. 
```
sh inference_vc.sh

# --ckpt "logs/hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth" \ LibriTTS-460
# --ckpt "logs/hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth" \ LibriTTS-960
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth" \ Large_v1 epoch 60 (paper version)
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \ Large_v1.1 epoch 200 (20. Nov. 2023)

CUDA_VISIBLE_DEVICES=0 python3 inference_vc.py \
                --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \
                --output_dir "vc_results_eng_kor_v2" \
                --noise_scale_vc "0.333" \
                --noise_scale_ttv "0.333" \
                --denoise_ratio "0"
```
- For better robustness, we recommend a noise_scale of 0.333
- For better expressiveness, we recommend a noise_scale of 0.667
- Find your best parameters for your style prompt 😵
- Voice Conversion is vulnerable to noisy target prompt so we recommend to utilize a denoiser with noisy prompt
- For noisy source speech, a wrong F0 may be extracted by YAPPT resulting in a quality degradation.


## Speech Super-resolution
- SpeechSR-24k and SpeechSR-48 are provided in TTS pipeline. If you want to use SpeechSR only, please refer inference_speechsr.py.
- If you change the output resolution, add this
```
--output_sr "48000" # Default
--output_sr "24000" # 
--output_sr "16000" # without super-resolution.
```
## Speech Denoising for Noise-free Speech Synthesis (Only used in Speaker Encoder during Inference)
- For denoised style prompt, we utilize a denoiser [(MP-SENet)](https://github.com/yxlu-0102/MP-SENet).
- When using a long reference audio, there is an out-of-memory issue with this model so we have a plan to learn a memory efficient speech denoiser in the future.
- If you have a problem, we recommend to use a clean reference audio or denoised audio before TTS pipeline or denoise the audio with cpu (but this will be slow😥). 

## TTV-v2 (WIP)
- TTV-v1 is a simple model which is very slightly modified from VITS. Although this simple TTV could synthesize a speech with high-quality and high speaker similarity, we thought that there is room for improvement in terms of expressiveness such as prosody modeling.
- For TTV-v2, we modify some components and training process (Model size: 107M --> 278M)
  1. Intermediate hidden size: 256 --> 384 
  2. Loss masking for wav2vec reconstruction loss (I left out masking the loss for zero-padding sequences😥)
  3. For long sentence generation, we finetune the model with full LibriTTS-train dataset without data filtering (Decrease the learning rate to 2e-5 with batch size of 8 per gpus)
  4. Multi-lingual Dataset (We are training the model with Eng, Indic, and Kor dataset now)    

## GAN VS Diffusion
<details> 
<summary> [Read More] </summary>
We think that we could not confirm which is better yet. There are many advatanges for each model so you can utilize each model for your own purposes and each study must be actively conducted simultaneously.  

### GAN (Specifically, GAN-based End-to-End Speech Synthesis Models)
- (pros) Fast Inference Speed 
- (pros) High-quality Audio
- (cons) Slow Training Speed (Over 7~20 Days)
- (cons) Lower Voice Style Transfer Performance than Diffusion Models
- (cons) Perceptually High-quality but Over-smoothed Audio because of Information Bottleneck by the sampling from the low-dimensional Latent Variable
   
### Diffusion (Diffusion-based Mel-spectrogram Generation Models)
- (pros) Fast Training Speed (within 3 Days)
- (pros) High-quality Voice Style Transfer
- (cons) Slow Inference Speed
- (cons) Lower Audio quality than End-to-End Speech Synthesis Models 

### (In this wors) Our Approaches for GAN-based End-to-End Speech Synthesis Models 
- Improving Voice Style Transfer Performance in End-to-End Speech Synthesis Models for OOD (Zero-shot Voice Style Transfer for Novel Speaker)
- Improving the Audio Quality beyond Perceptal Quality for Much more High-fidelity Audio Generation

### (Our other works) Diffusion-based Mel-spectrogram Generation Models
- DDDM-VC: Disentangled Denoising Diffusion Models for High-quality and High-diversity Speech Synthesis Models
- Diff-hierVC: Hierarhical Diffusion-based Speech Synthesis Model with Diffusion-based Pitch Modeling

### Our Goals
- Integrating each model for High-quality, High-diversity and High-fidelity Speech Synthesis Models 
</details> 

## LLM-based Models
We hope to compare LLM-based models for zero-shot TTS baselines. However, there is no public-available official implementation of LLM-based TTS models. Unfortunately, unofficial models have a poor performance in zero-shot TTS so we hope they will release their model for a fair comparison and reproducibility and for our speech community. THB I could not stand the inference speed almost 1,000 times slower than e2e models It takes 5 days to synthesize the full sentences of LibriTTS-test subsets. Even, the audio quality is so bad. I hope they will release their official source code soon. 

In my very personal opinion, VITS is still the best TTS model I have ever seen. But, I acknowledge that LLM-based models have much powerful potential for their creative generative performance from the large-scale dataset but not now.

## Limitation of our work
- Slow training speed and Relatively large model size (Compared with VITS) --> Future work: Light-weight and Fast training pipeline and much larger model...
- Could not generate realistic background sound --> Future work: adding audio generation part by disentangling speech and sound.
- Could not generate a speech from a too long sentence becauase of our training setting. We see increasing max length could improve the model performance. However, we do not have GPUs with 80 GB 😢 
  ```
   # Data Filtering for limited computation resource. 
    wav_min = 32
    wav_max = 600 # 12s 
    text_min = 1
    text_max = 200
  ```
TTV v2 may reduce this issue significantly...!

## Results [[Download]](https://drive.google.com/drive/folders/1xCrZQy9s5MT38RMQxKAtkoWUgxT5qYYW?usp=sharing)
We have attached all samples from LibriTTS test-clean and test-other. 

## Reference
<details> 
<summary> [Read More] </summary>
 
### Our Previous Works
- HierSpeech/HierSpeech-U for Hierarchical Speech Synthesis Framework: https://openreview.net/forum?id=awdyRVnfQKX
- HierVST for Baseline Speech Backbone: https://www.isca-speech.org/archive/interspeech_2023/lee23i_interspeech.html
- DDDM-VC: https://dddm-vc.github.io/
- Diff-HierVC: https://diff-hiervc.github.io/
  
### Baseline Model
- VITS: https://github.com/jaywalnut310/vits
- NaturalSpeech 
- NANSY for Audio Perturbation: https://github.com/revsic/torch-nansy
- Speech Resynthesis: https://github.com/facebookresearch/speech-resynthesis
  
### Waveform Generator for High-quality Audio Generation
- HiFi-GAN: https://github.com/jik876/hifi-gan 
- BigVGAN for High-quality Generator: https://arxiv.org/abs/2206.04658
- UnivNET: https://github.com/mindslab-ai/univnet
- EnCodec: https://github.com/facebookresearch/encodec

### Self-supervised Speech Model 
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- XLS-R: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- MMS: https://huggingface.co/facebook/facebook/mms-300m

### Other Large Language Model based Speech Synthesis Model
- VALL-E & VALL-E-X
- SPEAR-TTS
- NaturalSpeech 2
- Make-a-Voice
- MEGA-TTS & MEGA-TTS 2
- UniAudio

### AdaLN-zero
- Dit: https://github.com/facebookresearch/DiT
  
Thanks for all nice works. 
</details>

## LICENSE
- Code in this repo: MIT License
- Model Weights: CC-BY-NC-4.0 license
