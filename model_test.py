import os, argparse
from model_module import *

from model import *
from model_loss import *
from model_data import *

from complex_layers.STFT import *
from complex_layers.networks import *
from complex_layers.activations import *


speech_length = 16384
sampling_rate = 16000


'GET UNSEEN SPEECH FILE PATH'
def get_file_list (file_path):

      file_list = []

      for root, dirs, files in (os.walk(file_path)):
            for fname in files: 
                  if fname == "desktop.ini" or fname == ".DS_Store": continue 

                  full_fname = os.path.join(root, fname)
                  file_list.append(full_fname)

      file_list = natsort.natsorted(file_list, reverse = False)
      file_list = np.array(file_list)

      return file_list


'INFERENCE DEEP LEARNING MODEL'
def inference (path_list, save_path):

      for index1, speech_file_path in tqdm(enumerate (path_list)):
            _, unseen_noisy_speech = scipy.io.wavfile.read(speech_file_path)
            restore = []
            
            for index2 in range (int(len(unseen_noisy_speech) / speech_length)):
                  split_speech = unseen_noisy_speech[speech_length * index2 : speech_length * (index2 + 1)]
                  split_speech = np.reshape(split_speech, (1, speech_length, 1))
                  enhancement_speech = model.predict([split_speech])
                  predict = np.reshape(enhancement_speech, (speech_length, 1))
                  restore.extend(predict)
            restore = np.array(restore)
            scipy.io.wavfile.write("./model_pred/2000825_" + "{:4d}".format(index1+1) + ".wav", rate = sampling_rate, data = restore)



if __name__ == "__main__":

      parser = argparse.ArgumentParser(description = 'SETTING OPTION')
      parser.add_argument("--model", type = str, default = "dcunet16",         help = "Input model type")
      parser.add_argument("--load", type = str, default = "./model_save/model.h5", help = "Input save model file")
      parser.add_argument("--data",  type = str, default = "./datasets/unseen/",    help = "Input load unseen speech")
      parser.add_argument("--save",  type = str, default = "./model_pred/",      help = "Input save predict speech")
      args = parser.parse_args()

      model_type = args.model
      load_file_path = args.load
      test_data_path = args.data
      pred_data_path = args.save

      if model_type == "dcunet16":
            model = Naive_DCUnet16().model()
            model.summary()
            model.load_weights(load_file_path)
      elif model_type == 'dcunet20':
            model = Naive_DCUnet20().model()
            model.summary()
            model.load_weights(load_file_path)

      'READ SPEECH FILE'
      noisy_file_list = get_file_list(file_path = test_data_path)

      'INFERENCE'
      inference(path_list = noisy_file_list, save_path = pred_data_path)
      print("__END__")