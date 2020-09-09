# Introudction
Impelmentation Phase-aware Speech Enhnacement Deep Complex U Net  
This is convolution neural networks model for Speech Enhancement  
Papers URL
1. https://openreview.net/pdf?id=SkeRTsAcYm  
2. https://arxiv.org/abs/1903.03107  
#
# Requirements
Python >= 3.6.9  
numpy  
scipy  
librosa 0.7.2  
Tensorflow >= 2.1.0  
#  
# Directory  
```
Directory
./DCUnet
    /datasets
        /train_clean
        /train_noise
        /test_clean
        /test_noise
    /model_save
    /model_pred
    /complex_layers
        __init__.py
        STFT.py
        networks.py
        activation.py
        normaliztion.py
    model.py
    model_module.py
    model_loss.py
    model_data.py
    model_test.py
    model_train.py
```
#
# Usage
```
python model_train.py --model dcunet16 --trn ./datasets/subset_noisy/ --trc ./datasets/subset_clean/ --batch 16
python model_train.py --model dcunet20 --trn ./datasets/subset_noisy/ --trc ./datasets/subset_clean/ --batch 8


Not yet implementation
1. Optional SDR_loss, weighted_SDR_loss
2. 모델 검증
```
