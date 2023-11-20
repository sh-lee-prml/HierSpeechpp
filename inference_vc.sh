# --ckpt "logs/hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth" \ LibriTTS-460
# --ckpt "logs/hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth" \ LibriTTS-960
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth" \ Large_v1 epoch 60 (paper version)
# --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v2_ckpt.pth" \ Large_v1.1 epoch 200 (20. Nov. 2023)

CUDA_VISIBLE_DEVICES=0 python3 inference_vc.py \
                --ckpt "logs/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth" \
                --output_dir "vc_results_eng_kor_v2" \
                --noise_scale_vc "0.333" \
                --noise_scale_ttv "0.333" \
                --denoise_ratio "0"
