# Introudction
Impelmentation Phase-aware Speech Enhnacement Deep Complex U Net  
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
