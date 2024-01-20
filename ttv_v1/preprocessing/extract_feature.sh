#!/bin/bash
 
read -p "Enter the stage to process (1: w2v | 2: F0 | 3: txt): " stage
input_wav_dir='/workspace/raid/dataset/LibriTTS_16k'
input_txt_dir='/workspace/raid/dataset/LibriTTS_text'

# Stage 1: extract w2v feature from MMS
if [ "$stage" -eq 1 ]; then
    echo "Extracting w2v features..."
    python3 extract_w2v.py -i "$input_wav_dir" 
fi

# Stage 2: extract F0 using YAAPT
if [ "$stage" -eq 2 ]; then
    echo "Extracting F0..."
    python3 extract_f0.py -i "$input_wav_dir"
fi

# Stage 3: extract text token using espeak-phonemizer
if [ "$stage" -eq 3 ]; then
    echo "Extracting text tokens..."
    python3 extract_token.py -i "$input_txt_dir"
fi

echo "compledted $stage."
