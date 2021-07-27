import os
import argparse
from model_loss import *
from model_data import *
from model_module import *
from complex_layers import *
from complex_layers.stft import *
from complex_layers.layer import *
from complex_layers.activations import *

'PRINT SYSTEM INFORMATION'
print("GPU AVAILABLE", tf.config.list_physical_devices('GPU'))


def data_generator(train_arguments, test_arguments):
      train_generator = datagenerator(**train_arguments)
      test_generator  = datagenerator(**test_arguments)
      return train_generator, test_generator


@tf.function
def loop_train (model, optimizer, train_noisy_speech, train_clean_speech):
      with tf.GradientTape() as tape:
            train_predict_speech = model(train_noisy_speech)
            if loss_function == "SDR":
                  train_loss = modified_SDR_loss(train_predict_speech, train_clean_speech)
            elif loss_function == "wSDR":
                  train_loss = weighted_SDR_loss(train_noisy_speech, train_predict_speech, train_clean_speech)

      gradients = tape.gradient(train_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return train_loss


@tf.function
def loop_test (model, test_noisy_speech, test_clean_speech):
      'Test loop do not caclultae gradient and backpropagation'
      test_predict_speech = model(test_noisy_speech)
      if loss_function == "SDR":
            test_loss = modified_SDR_loss(test_predict_speech, test_clean_speech)
      elif loss_function == "wSDR":
            test_loss = weighted_SDR_loss(test_noisy_speech, test_predict_speech, test_clean_speech)
      return test_loss


def learning_rate_scheduler (epoch, learning_rate):
      if (epoch+1) <= int(0.5*epoch):
            return 1.00 * learning_rate
      elif (epoch+1) > int(0.5*epoch) and (epoch+1) <= int(0.75*epoch):
            return 0.20 * learning_rate
      else:
            return 0.05 * learning_rate
      

def model_flow (model, total_epochs, train_generator, test_generator):
      # DEFINE TRAIN STEP, TEST STEP
      train_step = len(os.listdir(train_noisy_path)) // batch_size
      test_step  = len(os.listdir(test_noisy_path)) // batch_size
      print("TRAIN STEPS, TEST STEPS   ", train_step, test_step)

      for epoch in tqdm(range (total_epochs)):
            train_batch_losses = 0
            test_batch_losses  = 0
            optimizer          = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler(epoch, learning_rate), beta_1 = 0.9)

            'Training Loop'
            for index, (train_noisy_speech, train_clean_speech) in tqdm(enumerate(train_generator)):
                  loss = loop_train (model, optimizer, train_noisy_speech, train_clean_speech)
                  train_batch_losses = train_batch_losses + loss

            'Test Loop'
            for index, (test_noisy_speech, test_clean_speech) in tqdm(enumerate(test_generator)):
                  loss  = loop_test (model, test_noisy_speech, test_clean_speech)
                  test_batch_losses  = test_batch_losses + loss

            'Calculate loss per batch data'
            train_loss = train_batch_losses / train_step
            test_loss  = test_batch_losses / test_step

            templet = "Epoch : {:3d},     TRAIN LOSS : {:.5f},     TEST LOSS  :  {:.5f}"
            print(templet.format(epoch+1, train_loss.numpy(), test_loss.numpy()))

            if ((epoch+1) % 10) == 0: 
                  model.save_weights("./model_save/" + save_file_name + str(epoch+1) + ".h5")



if __name__ == "__main__":
      tf.random.set_seed(seed = 42)
      parser = argparse.ArgumentParser(description = 'MODEL SETTING OPTION...')
      parser.add_argument("--model", type = str, default = "naive_dcunet20", help = "model type")
      parser.add_argument("--epoch", type = int, default = 200,        help = "Input epochs")
      parser.add_argument("--batch", type = int, default = 64,         help = "Input batch size")
      parser.add_argument("--loss",  type = str, default = "wSDR",     help = "Input Loss function")
      parser.add_argument("--optim", type = str, default = "adam",     help = "Input optimizer option")
      parser.add_argument("--lr",    type = float, default = 0.002,    help = "Inputs learning rate")
      parser.add_argument("--trn",   type = str, default = "./datasets/train_noisy/", help = "training noisy")
      parser.add_argument("--trc",   type = str, default = "./datasets/train_clean/", help = "training clean")
      parser.add_argument("--ten",   type = str, default = "./datasets/test_noisy/",  help = "testing noisy")
      parser.add_argument("--tec",   type = str, default = "./datasets/test_clean/",  help = "testing clean")
      parser.add_argument("--save",  type = str, default = "dcunet",                  help = "save model file name")
      args             = parser.parse_args()
      model_type       = args.model
      total_epochs     = args.epoch
      batch_size       = args.batch
      loss_function    = args.loss
      optimizer_type   = args.optim
      learning_rate    = args.lr
      train_noisy_path = args.trn
      train_clean_path = args.trc
      test_noisy_path  = args.ten
      test_clean_path  = args.tec
      save_file_name   = args.save

      train_arguments = {"inputs_ids" : os.listdir(train_noisy_path), 
                        "outputs_ids" : os.listdir(train_clean_path),
                        "inputs_dir" : train_noisy_path, 
                        "outputs_dir" : train_clean_path,
                        "batch_size" : batch_size}
      test_arguments  = {"inputs_ids" : os.listdir(test_noisy_path), 
                        "outputs_ids" : os.listdir(test_clean_path),
                        "inputs_dir" : test_noisy_path,
                        "outputs_dir" : test_clean_path,
                        "batch_size" : batch_size}
      train_generator, test_generator = data_generator(train_arguments = train_arguments, test_arguments = test_arguments)

      if model_type == "naive_dcunet16":
            model = Naive_DCUnet16().model()
      elif model_type == "naive_dcunet20":
            model = Naive_DCUnet20().model()
      elif model_type == "dcunet16":
            model = DCUnet16().model()
      elif model_type == "dcunet20":
            model = DCUnet20().model()
      
      model.summary()
      model_flow (model, total_epochs, train_generator, test_generator)
