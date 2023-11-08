# --ckpt "logs/hierspeechpp_libritts460/G_460000.pth" \ LibriTTS-460
# --ckpt "logs/hierspeechpp_libritts960/G_1230000.pth" \ LibriTTS-960
# --ckpt "logs/hierspeechpp_eng_kor/G_720000.pth" \ Large_v1
# --ckpt "logs/hierspeechpp_eng_kor/G_1340000.pth" \ Large_v2

CUDA_VISIBLE_DEVICES=1 python3 inference.py \
                --ckpt "logs/hierspeechpp_eng_kor/G_1380000.pth" \
                --ckpt_text2w2v "logs/ttv_libritts_v1/G_950000.pth" \
                --output_dir "tts_results_eng_kor_v2" \
                --noise_scale_vc "0.333" \
                --noise_scale_ttv "0.333" \
                --denoise_ratio "0"