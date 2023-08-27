# HierSpeech2: Hierarhical Variational Autoencoder is a Strong Zero-shot Speech Synthesizer 
The official implementation of HierSpeech2 | [Paper]() | [Demo page]()

**Sang-Hoon Lee, Ha-Yeong Choi, Seong-Whan Lee<sup>*</sup>**

Department of Artificial Intelligence, Korea University, Seoul, Korea  
<sup>*</sup> Corresponding author

## Abstract


## Previous Works
- [1] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis
- [2] HierVST: HierVST: Hierarchical Adaptive Zero-shot Voice Style Transfer

This paper is an extenstion version of above papers.

## Todo
- [ ] HierSpeech2-16k (Efficient but Strong Zero-shot Speech Synthesizier)
- [ ] HierSpeech2-16k-Large (For Much More Strong Zero-shot Speech Synthesizer)
- [ ] HierSpeech2-16k-Large-Full (For High-quality Cross-lingual Speech Synthesizer)
- [ ] HierSpeech2-24k-Large-Full (For High-resolution and High-quality Speech Synthesizer)
- [ ] HierSpeech2-48k-Large-Full (For Industrial-level High-resolution and High-quality Speech Synthesizer)
- [ ] Text-to-Vec (For Text-to-Speech)
- [ ] Text-to-Vec-Large (For Much More Expressive Text-to-Speech)

## Getting Started

### Pre-requisites

## Checkpoint
| Model |Sampling Rate|Params|Dataset |Checkpoint|
|------|:---:|:---:|:---:|:---:|
| HierSpeech2 |16 kHz|75M| LibriTTS (train-460) |-|
| HierSpeech2-Large|16 kHz|200M| LibriTTS (train-460)  |-|
| HierSpeech2-Large-Full|16 kHz|200M| LibriTTS (train-960)  |-|
| HierSpeech2-Large-Korean|16 kHz|200M| LibriTTS (train-960, NIKL, AudioBook-Korean)  |-|
| HierSpeech2-Large-Full|24 kHz|200M| Not Available |Not Available|
| HierSpeech2-Large-Full|48 kHz|200M| Not Available |Not Available|


## Voice Conversion
- Todo
- 
## Text-to-Speech
- Todo
- 
## F0 Extraction
- Todo
- 
## Neural Upsampling
- Todo
- 

## Reference
### Our Previous Works
- HierSpeech/HierSpeech-U for Hierarchical Speech Synthesis Framework: https://openreview.net/forum?id=awdyRVnfQKX
- HierVST for Baseline Speech Backbone: https://www.isca-speech.org/archive/interspeech_2023/lee23i_interspeech.html 
- DDDM-VC: https://dddm-vc.github.io/
- Diff-HierVC: https://diff-hiervc.github.io/
  
### Baseline Model
- VITS: https://github.com/jaywalnut310/vits
- NANSY for Audio Perturbation: https://github.com/revsic/torch-nansy
  
### Waveform Generator for High-quality Audio Generation
- BigVGAN for High-quality Generator: https://arxiv.org/abs/2206.04658
- UnivNET: https://github.com/mindslab-ai/univnet
- EnCodec: https://github.com/facebookresearch/encodec

### Self-supervised Speech Model 
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- XLS-R: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- MMS: https://huggingface.co/facebook/facebook/mms-300m

