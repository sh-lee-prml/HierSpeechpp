# HierSpeech2: Bridging the Gap between Semantic and Acoustic Representation by Hierarchical Variational Inference for Zero-shot Speech Synthesis
The official implementation of HierSpeech2 | [Paper]() | [Demo page]()

<!--
# HierSpeech3: Bridging the Gap between Text Prompt and Style Representation by Hierarchical Style Prompt Modeling for Zero-shot Voice and Prosody Generation
The official implementation of HierSpeech3 | [Paper]() | [Demo page]()
-->

**Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee**

 Department of Artificial Intelligence, Korea University, Seoul, Korea  

## Abstract

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

## Checkpoint
### Hierarchical Speech Synthesizer
| Model |Sampling Rate|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960) |555|2,311|-|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960), Libri-light (Small, Medium), Expresso, MMS(Kor), NIKL(Kor)|20k| 10,000 |60 epochs
(Paper ver.)|
| HierSpeech2|16 kHz|97M| LibriTTS (train-960), Libri-light (Small, Medium), Expresso, MMS(Kor), NIKL(Kor)|20k| 10,000 |120 epochs|

<!--
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960))  |-|
| HierSpeech2-Lite|16 kHz|-| LibriTTS (train-960) NIKL, AudioBook-Korean)  |-|
| HierSpeech2-Large-CL|16 kHz|200M| LibriTTS (train-960), Libri-Light, NIKL, AudioBook-Korean, Japanese, Chinese, CSS, MLS)  |-|
-->

### TTV
| Model |Language|Params|Dataset|Hour|Speaker|Checkpoint|
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| TTV |Eng|100M| LibriTTS (train-960) |555|2,311|-|
| TTV |Kor|100M| NIKL |114|118|-|

<!--
| TTV |Eng|50M| LibriTTS (train-960) |555|2,311|-|
| TTV-Large |Eng|100M| LibriTTS (train-960) |555|2,311|-|
| TTV-Lite |Eng|10M| LibriTTS (train-960) |555|2,311|-|
| TTV |Kor|50M| NIKL |114|118|-|
-->
### SpeechSR
| Model |Sampling Rate|Params|Dataset |Checkpoint|
|------|:---:|:---:|:---:|:---:|
| SpeechSR-24k |16kHz --> 24 kHz|0.03M| LibriTTS (train-960), MMS (Kor) |-|
| SpeechSR-48k |16kHz --> 48 kHz|0.05M| MMS (Kor), Singing (Kor), Expresso (Eng)|-|

## Voice Conversion
- Todo
- 
## Text-to-Speech
- Todo
- 
## Speech Super-resolution
- Todo

## Speech Denoising for Noise-free Speech Synthesis (Only used in Speaker Encoder during Inference)
- Todo

## GAN VS Diffusion

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

## LLM-based Models
We hope to compare LLM-based models for zero-shot TTS baselines. However, there is no public-available official implementation of LLM-based TTS models. Unfortunately, unofficial models have a poor performance in zero-shot TTS so we hope they will release their model for a fair comparison and reproducibility and for our speech community. THB I could not tolerate the inference speed of unofficial models above 2,000 times slower than e2e models It takes 5 days ㅡㅡ to synthesize the full sentences of LibriTTS-test subsets. Even, the audio quality is so so so s oso bad. I hope they will release their official source code soon. 

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

### Other TTS paper
- Tacotron2:
- FastSpeech:
- Glow-TTS:
- WaveNet:

Thanks for all nice works. 
