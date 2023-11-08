# HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation by Hierarchical Variational Inference for Zero-shot Speech Synthesis
The official implementation of HierSpeech2 | [Paper]() | [Demo page]() | 

**Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee**

 Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract
<details> 
<summary> [Abs.] Sorry for too long abstract😅 </summary>


Recently, large-scale language models (LLM)-based speech synthesis has shown a significant performance in zero-shot speech synthesis. However, they require a large-scale data and even suffer from the same limitation of previous autoregressive speech models such as slow inference speed and lack of robustness. Following the previous powerful end-to-end text-to-speech framework of VITS (but now that's what we call classical), this paper proposes HierSpeech++, a fast and strong zero-shot speech synthesizer for text-to-speech (TTS) and voice conversion (VC). In the previous our works (HierSpeech and HierVST), we verified that hierarchical speech synthesis frameworks could significantly improve the robustness and expressiveness of the synthetic speech by adopting hierarchical variational autoencoder and leveraging self-supervised speech represenation as an additional linguistic information to bridge an information gap between text and speech. In this work, we once again significantly improve the naturalness and speaker similarity of the synthetic speech even in the zero-shot speech synthesis scenarios. We first introduce multi-audio acoustic encoder for the enhanced acoustics posterior, and adopt a hierarchical adaptive waveform generator with conditional/unconditional generation. Second, we additionally utilize a F0 information and introduce source-filter theory-based multi-path semantic encoder for speaker-agnostic and speaker-related semantic representation. We also leverage hierarchical variational autoencoder to connect multiple representations, and present a BiT-Flow which is a bidirectional normalizing flow Transformer networks with AdaLN-Zero for better speaker adaptation and train-inference mismatch reduction. Without any text transcripts, we only utilize the speech dataset to train the speech synthesizer for data flexibility. For text-to-speech, we introduce text-to-vec (TTV) frameworks to generate a self-supervised speech representation and F0 representation from text representation and prosody prompt. Then, the speech synthesizer of HierSpeech++ generates the speech from generated vector, F0, and voice prompt. In addition, we propose the high-efficient speech super-resolution framework which can upsample the waveform audio from 16 kHz to 48 kHz, and this facilitate training the speech synthesizer in that we can use easily available low-resolution (16 kHz) speech data for scaling-up. The experimental results demonstrated that hierarchical variational autoencoder could be a strong zero-shot speech synthesizer by beating LLM-based models and diffusion-based models for TTS and VC tasks. Furthermore, we also verify the data efficiency in that our model trained with a small dataset still shows a better performance in both naturalness and similarity than other models trained with large-scale dataset. Moreover, we achieve the first human-level quality in zero-shot speech synthesis.
</details>

## Previous Our Works
- [1] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis
- [2] HierVST: Hierarchical Adaptive Zero-shot Voice Style Transfer

This paper is an extenstion version of above papers.

## Todo
### Hierarchical Speech Synthesizer
- [ ] HierSpeech2-Backbone
<!--
- [ ] HierSpeech-Lite (Fast and Efficient Zero-shot Speech Synthesizer)
- [ ] HierSinger (Zero-shot Singing Voice Synthesizer)
- [ ] HierSpeech2-24k-Large-Full (For High-resolutional and High-quality Speech Synthesizer)
- [ ] HierSpeech2-48k-Large-Full (For Industrial-level High-resolution and High-quality Speech Synthesizer)
-->
### Text-to-Vec (TTV)
- [ ] Text-to-Vec (LibriTTS-train-960)
- [ ] Multi-lingual Text-to-Vec (Will be released in Dec. 2023)
- [ ] Korean TTV (Will be released in 2024)
<!--
- [ ] Hierarchical Text-to-Vec (For Much More Expressive Text-to-Speech)
-->
### Speech Super-resolution (16k --> 24k or 48k) 
- [ ] SpeechSR-24k
- [ ] SpeechSR-48k

## Getting Started

### Pre-requisites

## Checkpoint [[Download]](https://drive.google.com/drive/folders/1-L_90BlCkbPyKWWHTUjt5Fsu3kz0du0w?usp=sharing)
### Hierarchical Speech Synthesizer
| Model |Sampling Rate|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HierSpeech2|16 kHz|97M| LibriTTS (train-460) |555|2,311|[[Download]](https://drive.google.com/drive/folders/14FTu0ZWux0zAD7ev4O1l6lKslQcdmebL?usp=sharing)|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960) |555|2,311|[[Download]](https://drive.google.com/drive/folders/1sFQP-8iS8z9ofCkE7szXNM_JEy4nKg41?usp=drive_link)|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960), Libri-light (Small, Medium), Expresso, MMS(Kor), NIKL(Kor)|20k| 10,000 |[[Download]](https://drive.google.com/drive/folders/14jaDUBgrjVA7bCODJqAEirDwRlvJe272?usp=drive_link)|

<!--
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960))  |-|
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960) NIKL, AudioBook-Korean)  |-|
| HierSpeech2-Large-CL|16 kHz|200M| LibriTTS (train-960), Libri-Light, NIKL, AudioBook-Korean, Japanese, Chinese, CSS, MLS)  |-|
-->

### TTV
| Model |Language|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| TTV |Eng|100M| LibriTTS (train-960) |555|2,311|[[Download]](https://drive.google.com/drive/folders/1QiFFdPhqhiLFo8VXc0x7cFHKXArx7Xza?usp=drive_link)|


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
| SpeechSR-24k |16kHz --> 24 kHz|0.13M| LibriTTS (train-960), MMS (Kor) |speechsr24k|
| SpeechSR-48k |16kHz --> 48 kHz|0.13M| MMS (Kor), Expresso (Eng), VCTK (Eng)|speechsr48k|

## Voice Conversion
- Todo
- 
## Text-to-Speech
```
sh inference.sh
```  
## Speech Super-resolution
- SpeechSR-24k and SpeechSR-48 are provided in TTS pipeline. If you want to use SpeechSR only, please refer [SpeechSR repository]().

## Speech Denoising for Noise-free Speech Synthesis (Only used in Speaker Encoder during Inference)
- For denoised style prompt, we utilize a denoiser [(MP-SENet)](https://github.com/yxlu-0102/MP-SENet).
- When using a long reference audio, there is an out-of-memory issue with this model so we have a plan to learn a memory efficient speech denoiser in the future.
- If you have a problem, we recommend to use a clean reference audio or denoised audio before TTS pipeline or denoise the audio with cpu (but this will be slow😥). 

## GAN VS Diffusion
<details> 
<summary> [Read Moro] </summary>
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
- SDT: Efficient Speech Diffusion Transformer Models

### Our Goals
- Integrating each model for High-quality, High-diversity and High-fidelity Speech Synthesis Models 
</details> 

## LLM-based Models
We hope to compare LLM-based models for zero-shot TTS baselines. However, there is no public-available official implementation of LLM-based TTS models. Unfortunately, unofficial models have a poor performance in zero-shot TTS so we hope they will release their model for a fair comparison and reproducibility and for our speech community. THB I could not tolerate the inference speed of unofficial models above 2,000 times slower than e2e models It takes 5 days to synthesize the full sentences of LibriTTS-test subsets. Even, the audio quality is so bad. I hope they will release their official source code soon. 

In my very personal opinion, VITS is still the best TTS model I have ever seen. But, I acknowledge that LLM-based models have much powerful potential for their creative generative performance from the large-scale dataset but not now.

## Limitation of our work
- Slow training speed and Relatively large model size (Compared with VITS) --> Future work: Much larger model????, Light-weight and Fast training pipeline
- Could not generate realistic background sound --> Future work: adding audio generation part by disentangling speech and sound. 

## Reference
### Our Previous Works
- HierSpeech/HierSpeech-U for Hierarchical Speech Synthesis Framework: https://openreview.net/forum?id=awdyRVnfQKX
- HierVST for Baseline Speech Backbone: https://www.isca-speech.org/archive/interspeech_2023/lee23i_interspeech.html
- DDDM-VC: https://dddm-vc.github.io/
- Diff-HierVC: https://diff-hiervc.github.io/
- SDT: https://sdt.github.io/

  
### Baseline Model
- VITS: https://github.com/jaywalnut310/vits
- NaturalSpeech: 
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
- VALL-E & VALL-E-X:
- SPEAR-TTS:
- NaturalSpeech 2: 
- Make-a-Voice:
- MEGA-TTS & MEGA-TTS 2:
- UniAudio: 

### Other TTS paper
- Tacotron2:
- FastSpeech:
- Glow-TTS:
- WaveNet:

Thanks for all nice works. 
