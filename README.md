# HierSpeech2: Hierarhical Variational Autoencoder is a Strong Zero-shot Unified Speech Synthesizer 
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
```- [ ] Text-to-Vec-Large-ML (For Multi-lingual Text-to-Speech)```

## Getting Started

### Pre-requisites

## Checkpoint
| Model |Sampling Rate|Dataset |Checkpoint|
|------|:---:|:---:|:---:|
| HierSpeech2 |16 kHz| LibriTTS (train-clean-360, train-clean-100) |-|
| HierSpeech2-Large|16 kHz| LibriTTS (train-clean-360, train-clean-100)  |-|
| HierSpeech2-Large-Full|16 kHz| LibriTTS (train-clean-360, train-clean-100, train-other-500, VCTK, CSS10, NIKL, Others)  |-|
| HierSpeech2-Large|24 kHz| Not Available |-|

## Reference
- HierSpeech:
- HierVST:
- DDDM-VC:
- Diff-HierVC:
- VITS: https://github.com/jaywalnut310/vits
- UnivNET: https://github.com/mindslab-ai/univnet
- Wav2Vec 2.0: https://arxiv.org/abs/2006.11477
- XLS-R: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- MMS:  
- BigVGAN: https://arxiv.org/abs/2206.04658
- NANSY: 
