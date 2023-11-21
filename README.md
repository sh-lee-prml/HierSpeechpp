# HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation by Hierarchical Variational Inference for Zero-shot Speech Synthesis <br><sub>The official implementation of HierSpeech++</sub>
## [Paper (Will be uploaded tomorrow)]() | [Demo page](https://sh-lee-prml.github.io/HierSpeechpp-demo/) | [Checkpoint](https://drive.google.com/drive/folders/1-L_90BlCkbPyKWWHTUjt5Fsu3kz0du0w?usp=sharing) | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/HierSpeech/HierSpeech_TTS)

**Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee**

 Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract
Large language models (LLM)-based speech synthesis has been widely adopted in zero-shot speech synthesis. However, they require a large-scale data and possess the same limitations as previous autoregressive speech models, including slow inference speed and lack of robustness. This paper proposes HierSpeech++, a fast and strong zero-shot speech synthesizer for text-to-speech (TTS) and voice conversion (VC). We verified that hierarchical speech synthesis frameworks could significantly improve the robustness and expressiveness of the synthetic speech. Furthermore, we significantly improve the naturalness and speaker similarity of synthetic speech even in zero-shot speech synthesis scenarios. For text-to-speech, we adopt the text-to-vec framework, which generates a self-supervised speech representation and an F0 representation based on text representations and prosody prompts. Then, HierSpeech++ generates speech from the generated vector, F0, and voice prompt. We further introduce a high-efficient speech super-resolution framework from 16 kHz to 48 kHz. The experimental results demonstrated that the hierarchical variational autoencoder could be a strong zero-shot speech synthesizer given that it outperforms LLM-based and diffusion-based models. Moreover, we achieved the first human-level quality zero-shot speech synthesis.

![Fig1_pipeline](https://github.com/sh-lee-prml/HierSpeechpp/assets/56749640/8f0b5f24-8491-4908-ae06-e0dfcc7d9e52)


This repository contains:

- 🪐 A PyTorch implementation of HierSpeech++ (TTV, Hierarchical Speech Synthesizer, SpeechSR)
- ⚡️ Pre-trained HierSpeech++ models trained on LibriTTS (Train-460, Train-960, and more dataset)
- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/HierSpeech/HierSpeech_TTS) Gradio Demo on HuggingFace. HuggingFace provides us with a community GPU grant. Thanks 😊
<!--
- 💥 A Colab notebook for running pre-trained HierSpeech++ models (Soon..)
🛸 A HierSpeech++ training script (Will be released soon)
-->
## Previous Our Works
- [NeurIPS2022] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis
- [Interspeech2023] HierVST: Hierarchical Adaptive Zero-shot Voice Style Transfer

This paper is an extenstion version of above papers.

## Todo
### Hierarchical Speech Synthesizer
- [x] HierSpeechpp-Backbone (LibriTTS-train-460)
- [x] HierSpeechpp-Backbone (LibriTTS-train-960)
- [x] HierSpeechpp-Backbone-60epoch (LibriTTS-train-960, Libri-light (Medium), Expresso, MSSS(Kor), NIKL(Kor))
- [x] HierSpeechpp-Backbone-200epoch (LibriTTS-train-960, Libri-light (Medium), Expresso, MSSS(Kor), NIKL(Kor))
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

### Cleaning Up the Source Code
- [ ] Clean Code
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
| HierSpeech2|16 kHz|97M| LibriTTS (train-960), Libri-light (Small, Medium), Expresso, MSSS(Kor), NIKL(Kor)|2,796| 7,299 |[[Download]](https://drive.google.com/drive/folders/14jaDUBgrjVA7bCODJqAEirDwRlvJe272?usp=drive_link)|

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
| SpeechSR-24k |16kHz --> 24 kHz|0.13M| LibriTTS (train-960), MSSS (Kor) |[speechsr24k](https://github.com/sh-lee-prml/HierSpeechpp/blob/main/speechsr24k/G_340000.pth)|
| SpeechSR-48k |16kHz --> 48 kHz|0.13M| MSSS (Kor), Expresso (Eng), VCTK (Eng)|[speechsr48k](https://github.com/sh-lee-prml/HierSpeechpp/blob/main/speechsr48k/G_100000.pth)|

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
# Mixup (Recommend 0.6~0.8)
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
- (21, Nov. 2023) Sliced window denoising. This may reduce a burden for denoising a speech.
  ```
        if denoise == 0:
            audio = torch.cat([audio.cuda(), audio.cuda()], dim=0)
        else:
            with torch.no_grad():
                
                if ori_prompt_len > 80000:
                    denoised_audio = []
                    for i in range((ori_prompt_len//80000)):
                        denoised_audio.append(denoise(audio.squeeze(0).cuda()[i*80000:(i+1)*80000], denoiser, hps_denoiser))
                    
                    denoised_audio.append(denoise(audio.squeeze(0).cuda()[(i+1)*80000:], denoiser, hps_denoiser))
                    denoised_audio = torch.cat(denoised_audio, dim=1)
                else:
                    denoised_audio = denoise(audio.squeeze(0).cuda(), denoiser, hps_denoiser)

            audio = torch.cat([audio.cuda(), denoised_audio[:,:audio.shape[-1]]], dim=0)
  ``` 

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
- Could not generate a speech from a too long sentence becauase of our training setting. We see increasing max length could improve the model performance. I hope to use GPUs with 80 GB 😢 
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
Our repository is heavily based on [VITS](https://github.com/jaywalnut310/vits) and [BigVGAN](https://github.com/NVIDIA/BigVGAN). 

<details> 
<summary> [Read More] </summary>
 
### Our Previous Works
- HierSpeech/HierSpeech-U for Hierarchical Speech Synthesis Framework: https://openreview.net/forum?id=awdyRVnfQKX
- HierVST for Baseline Speech Backbone: https://www.isca-speech.org/archive/interspeech_2023/lee23i_interspeech.html
- DDDM-VC: https://dddm-vc.github.io/
- Diff-HierVC: https://diff-hiervc.github.io/
  
### Baseline Model
- VITS: https://github.com/jaywalnut310/vits
- NaturalSpeech: https://speechresearch.github.io/naturalspeech/ 
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
- Make-a-Voice
- MEGA-TTS & MEGA-TTS 2
- UniAudio

### Diffusion-based Model
- NaturalSpeech 2

### AdaLN-zero
- Dit: https://github.com/facebookresearch/DiT
  
Thanks for all nice works. 
</details>
