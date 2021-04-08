from model_module import *


class Naive_DCUnet16 ():
    def __init__ (self, input_size = 16384, length = 1023, over_lapping = 256, padding = "same", norm_trainig = True):
        self.input_size   = input_size
        self.length       = length
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"window_length" : self.length, "over_lapping" : self.over_lapping, "padding" : self.padding}

    
    def model (self):
        noisy_speech         = Input (shape = (self.input_size, 1), name = "noisy_speech")
        stft_real, stft_imag = STFT_network(**self.STFT_network_arguments)(noisy_speech)
        stft_real, stft_imag = tranposed_STFT(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = encoder_module(stft_real, stft_imag, 32, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real2, conv_imag2 = encoder_module(real, imag, 32, (7, 5), (2, 1), training = self.norm_trainig)
        real, imag, conv_real3, conv_imag3 = encoder_module(real, imag, 64, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real4, conv_imag4 = encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real5, conv_imag5 = encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real6, conv_imag6 = encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real7, conv_imag7 = encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, _, _                   = encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        center_real1, center_imag1 = decoder_module(real, imag, None, None, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(center_real1, center_imag1, conv_real7, conv_imag7, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real6, conv_imag6, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real5, conv_imag5, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real4, conv_imag4, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real3, conv_imag3, 32, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real2, conv_imag2, 32, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real1, conv_imag1, 1, (5, 3), (2, 2), training = self.norm_trainig)

        enhancement_stft_real, enhancement_stft_imag = mask_processing(deconv_real1, deconv_imag1, stft_real, stft_imag)
        enhancement_stft_real, enhancement_stft_imag = transpoed_ISTFT(enhancement_stft_real, enhancement_stft_imag)

        enhancement_speech = ISTFT_network(**self.STFT_network_arguments)(enhancement_stft_real, enhancement_stft_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))

        return Model(inputs = [noisy_speech], outputs = [enhancement_speech])


class Naive_DCUnet20 ():
    def __init__ (self, input_size = 16384, length = 1023, over_lapping = 256, padding = "same", norm_trainig = True):
        self.input_size   = input_size
        self.length       = length
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"window_length" : self.length, "over_lapping" : self.over_lapping, "padding" : self.padding}

    
    def model (self):
        noisy_speech         = Input (shape = (self.input_size, 1), name = "noisy_speech")
        stft_real, stft_imag = STFT_network(**self.STFT_network_arguments)(noisy_speech)
        stft_real, stft_imag = tranposed_STFT(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = encoder_module(stft_real, stft_imag, 32, (7, 1), (1, 1), training = self.norm_trainig)
        real, imag, conv_real2, conv_imag2 = encoder_module(real, imag, 32, (1, 7), (1, 1), training = self.norm_trainig)
        real, imag, conv_real3, conv_imag3 = encoder_module(real, imag, 64, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real4, conv_imag4 = encoder_module(real, imag, 64, (7, 5), (2, 1), training = self.norm_trainig)
        real, imag, conv_real5, conv_imag5 = encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real6, conv_imag6 = encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real7, conv_imag7 = encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real8, conv_imag8 = encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real9, conv_imag9 = encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, _, _                   = encoder_module(real, imag, 90, (5, 3), (2, 1), training = self.norm_trainig)
        center_real1, center_imag1 = decoder_module(real, imag, None, None, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(center_real1, center_imag1, conv_real9, conv_imag9, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real8, conv_imag8, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real7, conv_imag7, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real6, conv_imag6, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real5, conv_imag5, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real4, conv_imag4, 64, (7, 5), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real3, conv_imag3, 32, (7, 5), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real2, conv_imag2, 32, (1, 7), (1, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real1, conv_imag1, 1, (7, 1), (1, 1), training = self.norm_trainig)

        enhancement_stft_real, enhancement_stft_imag = mask_processing(deconv_real1, deconv_imag1, stft_real, stft_imag)
        enhancement_stft_real, enhancement_stft_imag = transpoed_ISTFT(enhancement_stft_real, enhancement_stft_imag)

        enhancement_speech = ISTFT_network(**self.STFT_network_arguments)(enhancement_stft_real, enhancement_stft_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))
        return Model(inputs = [noisy_speech], outputs = [enhancement_speech])



class DCUnet16 ():
    def __init__ (self, input_size = 16384, length = 1023, over_lapping = 256, padding = "same", norm_trainig = True):
        self.input_size   = input_size
        self.length       = length
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"window_length" : self.length, "over_lapping" : self.over_lapping, "padding" : self.padding}

    
    def model (self):
        noisy_speech         = Input (shape = (self.input_size, 1), name = "noisy_speech")
        stft_real, stft_imag = STFT_network(**self.STFT_network_arguments)(noisy_speech)
        stft_real, stft_imag = tranposed_STFT(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = convariance_encoder_module(stft_real, stft_imag, 32, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real2, conv_imag2 = convariance_encoder_module(real, imag, 32, (7, 5), (2, 1), training = self.norm_trainig)
        real, imag, conv_real3, conv_imag3 = convariance_encoder_module(real, imag, 64, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real4, conv_imag4 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real5, conv_imag5 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real6, conv_imag6 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real7, conv_imag7 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, _, _                   = convariance_encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        center_real1, center_imag1 = convariance_decoder_module(real, imag, None, None, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(center_real1, center_imag1, conv_real7, conv_imag7, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real6, conv_imag6, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real5, conv_imag5, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real4, conv_imag4, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real3, conv_imag3, 32, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real2, conv_imag2, 32, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real1, conv_imag1, 1, (5, 3), (2, 2), training = self.norm_trainig)

        enhancement_stft_real, enhancement_stft_imag = mask_processing(deconv_real1, deconv_imag1, stft_real, stft_imag)
        enhancement_stft_real, enhancement_stft_imag = transpoed_ISTFT(enhancement_stft_real, enhancement_stft_imag)

        enhancement_speech = ISTFT_network(**self.STFT_network_arguments)(enhancement_stft_real, enhancement_stft_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))

        return Model(inputs = [noisy_speech], outputs = [enhancement_speech])



class DCUnet20 ():
    def __init__ (self, input_size = 16384, length = 1023, over_lapping = 256, padding = "same", norm_trainig = True):
        self.input_size   = input_size
        self.length       = length
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"window_length" : self.length, "over_lapping" : self.over_lapping, "padding" : self.padding}

    
    def model (self):
        noisy_speech         = Input (shape = (self.input_size, 1), name = "noisy_speech")
        stft_real, stft_imag = STFT_network(**self.STFT_network_arguments)(noisy_speech)
        stft_real, stft_imag = tranposed_STFT(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = convariance_encoder_module(stft_real, stft_imag, 32, (7, 1), (1, 1), training = self.norm_trainig)
        real, imag, conv_real2, conv_imag2 = convariance_encoder_module(real, imag, 32, (1, 7), (1, 1), training = self.norm_trainig)
        real, imag, conv_real3, conv_imag3 = convariance_encoder_module(real, imag, 64, (7, 5), (2, 2), training = self.norm_trainig)
        real, imag, conv_real4, conv_imag4 = convariance_encoder_module(real, imag, 64, (7, 5), (2, 1), training = self.norm_trainig)
        real, imag, conv_real5, conv_imag5 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real6, conv_imag6 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real7, conv_imag7 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, conv_real8, conv_imag8 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 1), training = self.norm_trainig)
        real, imag, conv_real9, conv_imag9 = convariance_encoder_module(real, imag, 64, (5, 3), (2, 2), training = self.norm_trainig)
        real, imag, _, _                   = convariance_encoder_module(real, imag, 90, (5, 3), (2, 1), training = self.norm_trainig)
        center_real1, center_imag1 = convariance_decoder_module(real, imag, None, None, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(center_real1, center_imag1, conv_real9, conv_imag9, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real8, conv_imag8, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real7, conv_imag7, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real6, conv_imag6, 64, (5, 3), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real5, conv_imag5, 64, (5, 3), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real4, conv_imag4, 64, (7, 5), (2, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real3, conv_imag3, 32, (7, 5), (2, 2), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real2, conv_imag2, 32, (1, 7), (1, 1), training = self.norm_trainig)
        deconv_real1, deconv_imag1 = convariance_decoder_module(deconv_real1, deconv_imag1, conv_real1, conv_imag1, 1, (7, 1), (1, 1), training = self.norm_trainig)

        enhancement_stft_real, enhancement_stft_imag = mask_processing(deconv_real1, deconv_imag1, stft_real, stft_imag)
        enhancement_stft_real, enhancement_stft_imag = transpoed_ISTFT(enhancement_stft_real, enhancement_stft_imag)

        enhancement_speech = ISTFT_network(**self.STFT_network_arguments)(enhancement_stft_real, enhancement_stft_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))

        return Model(inputs = [noisy_speech], outputs = [enhancement_speech])


if __name__ == "__main__":
    model = DCUnet20().model()
    model.summary()
    del model
    model = DCUnet16().model()
    model.summary()
