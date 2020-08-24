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
    /model_result
    /complex_layers
        __init__.py
        STFT.py
        complex_networks.py
        complex_activations.py
    module.py
    model.py
    model_loss.py
    model_test.py
    model_train.py
    datagenerator.py
```
#
# Usage
```
python model_train.py --epoch --batch --optim --model --train_noisy --train_clean --test_noisy --test_clean

--optim option is class SDR_Loss class weighted_SDR_Loss
--model option is class DCUnet_16 class DCUnet_20

Not yet complete implementation
class Complex_BatchNormalization (using Covariance)
```
