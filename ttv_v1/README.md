# Text-To-Vec (TTV)

### ✔️ Step 1: Extract features.

Extract each feature using `./ttv_v1/preprocessing/extract_feature.sh` in the preprocessing directory.
```
|-- preprocessing
|    |-- extract_f0.py
|    |-- extract_token.py
|    |-- extract_w2v.py
|    `-- extract_feature.sh
```

### ✔️ Step 2: Filter data and make a filelists. 

We filter by limiting the length of wav and text as follows:
```
wav_min = 32
wav_max = 600 # 12s 
text_min = 1
text_max = 200
```

For example, through the above `Step 1`, we can obtain the following data paths.
```
wave data path: '/workspace/raid/dataset/LibriTTS_16k/train-clean-100'
text data path: '/workspace/raid/dataset/LibriTTS_txt/train-clean-100'
F0 data path: '/workspace/raid/dataset/LibriTTS_f0/train-clean-100'
w2v data path: '/workspace/raid/dataset/LibriTTS_w2v/train-clean-100'
```

Modify the arguments appropriately for use.
```
parser.add_argument('-i', '--input_dir', default='/workspace/raid/dataset/LibriTTS_16k/train-clean-100')
parser.add_argument('-o', '--output_dir', default='/workspace/ha0/data_preprocess/filelist') 
```

 
By running `./ttv_v1/preprocessing/prepare_filelist.py`, a filelist can be created with the filtered data. 
 
```
python3 prepare_filelist.py
```

The list of files needed for training is as follows. Each .txt file contains the file path.
```
|-- filelist 
|    |-- train_f0.txt
|    |-- train_token.txt
|    |-- train_w2v.txt
|    `-- train_wav.txt
```


### ✔️ Step 3: Modify the configuration and train the model. 
Run `train_ttv_v1.py` as shown below. `-c` is the config path, `-m` is the name of the training model, and the model is saved in `./logs`.
```
CUDA_VISIBLE_DEVICES=0 python3 train_ttv_v1.py -c ttv_v1/config.json -m TTV_model
```

