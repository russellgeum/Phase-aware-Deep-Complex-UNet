from model_module import *



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'Simply SDR Loss'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def SDR_loss (pred_speech, true_speech, eps = 1e-8):

    num = K.sum(pred_speech * true_speech)
    den = K.sqrt(K.sum(true_speech * true_speech)) * K.sqrt(K.sum(pred_speech * pred_speech))

    return -(num / (den + eps))



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'Weighted SDR Loss'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def weighted_SDR_loss (noisy_speech, pred_speech, true_speech):
        
    noisy_speech = K.flatten(noisy_speech)
    pred_speech  = K.flatten(pred_speech)
    true_speech  = K.flatten(true_speech)

    pred_noise = noisy_speech - pred_speech
    true_noise = noisy_speech - true_speech
    
    def SDR_loss(pred, true, eps = 1e-8):

        num = K.sum(pred * true)
        den = K.sqrt(K.sum(true * true)) * K.sqrt(K.sum(pred * pred))
        
        return -(num / (den + eps))
        
    sound_SDR = SDR_loss(pred_speech, true_speech)
    noise_SDR = SDR_loss(pred_noise, true_noise)
    alpha = K.sum(true_speech**2) / (K.sum(true_speech**2) + K.sum(true_noise**2))

    return alpha * sound_SDR + (1-alpha) * noise_SDR