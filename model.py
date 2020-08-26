from model_module import *



class Naive_DCUnet_16 (tf.keras.Model):


    def __init__ (self, input_size = 16384, legnth = 1023, over_lapping = 256, padding = "same", norm_trainig = True):

        super(Naive_DCUnet_16, self).__init__()

        self.input_size   = input_size
        self.legnth       = legnth
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"length" : self.legnth, "over_lapping" : self.over_lapping, "padding" : self.padding}

        self.filter32 = 32
        self.filter64 = 64
        self.kernel_size75 = (7, 5)
        self.kernel_size53 = (5, 3)
        self.strides22 = (2, 2)
        self.strides21 = (2, 1)

        self.forward_STFT = STFT_network(**self.STFT_network_arguments)
        self.Inverse_STFT = ISTFT_network(**self.STFT_network_arguments)
        
        self.encoder1 = encoder(filters = self.filter32, kernel_size = self.kernel_size75, strides = self.strides22)
        self.encoder2 = encoder(filters = self.filter32, kernel_size = self.kernel_size75, strides = self.strides21)
        self.encoder3 = encoder(filters = self.filter64, kernel_size = self.kernel_size75, strides = self.strides22)

        self.encoder4 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.encoder5 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.encoder6 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.encoder7 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.encoder8 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)

        self.decoder1 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.decoder2 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.decoder3 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.decoder4 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.decoder5 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)

        self.decoder6 = decoder(filters = self.filter32, kernel_size = self.kernel_size75, strides = self.strides22)
        self.decoder7 = decoder(filters = self.filter32, kernel_size = self.kernel_size75, strides = self.strides21)
        self.decoder8 = decoder(filters = 1, kernel_size = self.kernel_size75, strides = self.strides22)


    def call (self, noisy_speech):

        stft_real, stft_imag = self.forward_STFT(noisy_speech)
        stft_real, stft_imag = convert2image(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = self.encoder1(stft_real, stft_imag, True)
        real, imag, conv_real2, conv_imag2 = self.encoder2(real, imag, True)
        real, imag, conv_real3, conv_imag3 = self.encoder3(real, imag, True)

        real, imag, conv_real4, conv_imag4 = self.encoder4(real, imag, True)
        real, imag, conv_real5, conv_imag5 = self.encoder5(real, imag, True)
        real, imag, conv_real6, conv_imag6 = self.encoder6(real, imag, True)
        real, imag, conv_real7, conv_imag7 = self.encoder7(real, imag, True)
        real, imag, _, _ = self.encoder8(real, imag, True)

        real, imag = self.decoder1(real, imag, None, None, training = True)
        real, imag = self.decoder2(real, imag, conv_real7, conv_imag7, True)
        real, imag = self.decoder3(real, imag, conv_real6, conv_imag6, True)
        real, imag = self.decoder4(real, imag, conv_real5, conv_imag5, True)
        real, imag = self.decoder5(real, imag, conv_real4, conv_imag4, True)

        real, imag = self.decoder6(real, imag, conv_real3, conv_imag3, True)
        real, imag = self.decoder7(real, imag, conv_real2, conv_imag2, True)
        real, imag = self.decoder8(real, imag, conv_real1, conv_imag1, True)

        enhancement_real, enhancement_imag = mask_processing(real, imag, stft_real, stft_imag)

        enhancement_real, enhancement_imag = convert2sequnce(enhancement_real, enhancement_imag)
        enhancement_speech = self.Inverse_STFT(enhancement_real, enhancement_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))

        return enhancement_speech




class Naive_DCUnet_20 (tf.keras.Model):


    def __init__ (self, input_size = 16384, legnth = 1023, over_lapping = 256, padding = "same", norm_trainig = True):

        super(Naive_DCUnet_20, self).__init__()

        self.input_size   = input_size
        self.legnth       = legnth
        self.over_lapping = over_lapping
        self.padding      = padding
        self.norm_trainig = norm_trainig
        self.STFT_network_arguments = {"length" : self.legnth, "over_lapping" : self.over_lapping, "padding" : self.padding}

        self.filter32 = 32
        self.filter64 = 64
        self.filter90 = 90
        self.kernel_size71 = (7, 1)
        self.kernel_size17 = (1, 7)
        self.kernel_size75 = (7, 5)
        self.kernel_size53 = (5, 3)
        self.strides11 = (1, 1)
        self.strides22 = (2, 2)
        self.strides21 = (2, 1)

        self.forward_STFT = STFT_network(**self.STFT_network_arguments)
        self.Inverse_STFT = ISTFT_network(**self.STFT_network_arguments)
        
        self.encoder1 = encoder(filters = self.filter32, kernel_size = self.kernel_size71, strides = self.strides11)
        self.encoder2 = encoder(filters = self.filter32, kernel_size = self.kernel_size17, strides = self.strides11)

        self.encoder3 = encoder(filters = self.filter64, kernel_size = self.kernel_size75, strides = self.strides22)
        self.encoder4 = encoder(filters = self.filter64, kernel_size = self.kernel_size75, strides = self.strides21)

        self.encoder5 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.encoder6 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.encoder7 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.encoder8 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.encoder9 = encoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.encoder10 = encoder(filters = self.filter90, kernel_size = self.kernel_size53, strides = self.strides21)

        self.decoder1 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.decoder2 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.decoder3 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.decoder4 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)
        self.decoder5 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides21)
        self.decoder6 = decoder(filters = self.filter64, kernel_size = self.kernel_size53, strides = self.strides22)

        self.decoder7 = decoder(filters = self.filter64, kernel_size = self.kernel_size75, strides = self.strides21)
        self.decoder8 = decoder(filters = self.filter32, kernel_size = self.kernel_size75, strides = self.strides22)

        self.decoder9 = decoder(filters = self.filter32, kernel_size = self.kernel_size17, strides = self.strides11)
        self.decoder10 = decoder(filters = 1, kernel_size = self.kernel_size71, strides = self.strides11)


    def call (self, noisy_speech):

        stft_real, stft_imag = self.forward_STFT(noisy_speech)
        stft_real, stft_imag = convert2image(stft_real, stft_imag)

        real, imag, conv_real1, conv_imag1 = self.encoder1(stft_real, stft_imag, True)
        real, imag, conv_real2, conv_imag2 = self.encoder2(real, imag, True)
        real, imag, conv_real3, conv_imag3 = self.encoder3(real, imag, True)
        real, imag, conv_real4, conv_imag4 = self.encoder4(real, imag, True)

        real, imag, conv_real5, conv_imag5 = self.encoder5(real, imag, True)
        real, imag, conv_real6, conv_imag6 = self.encoder6(real, imag, True)
        real, imag, conv_real7, conv_imag7 = self.encoder7(real, imag, True)
        real, imag, conv_real8, conv_imag8 = self.encoder8(real, imag, True)
        real, imag, conv_real9, conv_imag9 = self.encoder9(real, imag, True)
        real, imag, _, _ = self.encoder10(real, imag, True)

        real, imag = self.decoder1(real, imag, None, None, True)
        real, imag = self.decoder2(real, imag, conv_real9, conv_imag9, True)
        real, imag = self.decoder3(real, imag, conv_real8, conv_imag8, True)
        real, imag = self.decoder4(real, imag, conv_real7, conv_imag7, True)
        real, imag = self.decoder5(real, imag, conv_real6, conv_imag6, True)
        real, imag = self.decoder6(real, imag, conv_real5, conv_imag5, True)

        real, imag = self.decoder7(real, imag, conv_real4, conv_imag4, True)
        real, imag = self.decoder8(real, imag, conv_real3, conv_imag3, True)
        real, imag = self.decoder9(real, imag, conv_real2, conv_imag2, True)
        real, imag = self.decoder10(real, imag, conv_real1, conv_imag1, True)

        enhancement_real, enhancement_imag = mask_processing(real, imag, stft_real, stft_imag)

        enhancement_real, enhancement_imag = convert2sequnce(enhancement_real, enhancement_imag)
        enhancement_speech = self.Inverse_STFT(enhancement_real, enhancement_imag)
        enhancement_speech = tf.reshape(enhancement_speech, (-1, self.input_size, 1))

        return enhancement_speech


if __name__ == "__main__":
    
    noisy = Input(shape = (16384, 1), name = "noisy_speech")
    DCUnet16 = Naive_DCUnet_16()
    DCUnet20 = Naive_DCUnet_20()
    DCUnet16(noisy)
    DCUnet20(noisy)
    DCUnet16.summary()
    DCUnet20.summary()