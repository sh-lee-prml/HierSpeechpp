# Text-To-Vec (TTV)

### Guide

1. The list of files needed for training is as follows. Each .txt file contains the file path.
```
|-- filelist 
|    |-- train_f0.txt
|    |-- train_token.txt
|    |-- train_w2v.txt
|    `-- train_wav.txt
```

2. Extract each feature using `extract_feature.sh` in the preprocessing directory.
```
|-- preprocessing
|    |-- extract_f0.py
|    |-- extract_token.py
|    |-- extract_w2v.py
|    `-- extract_feature.sh
```

3. Run `train_ttv_v1.py` as shown below. `-c` is the config path, `-m` is the name of the training model, and the model is saved in `./logs`.
```
CUDA_VISIBLE_DEVICES=0 python3 train_ttv_v1.py -c ttv_v1/config.json -m TTV_model
```
