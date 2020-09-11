import os
import argparse
from model import *
from model_loss import *
from model_data import *
from model_module import *

from complex_layers.STFT import *
from complex_layers.networks import *
from complex_layers.activations import *


'PRINT SYSTEM INFORMATION'
print(tf.config.list_physical_devices())
print("GPU Available: ", tf.test.is_gpu_available('GPU'))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DATA_GENERATOR
LOOP_TRAIN
LOOP_TEST
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def data_generator(train_arguments, test_arguments):

      train_generator = datagenerator(**train_arguments)
      test_generator  = datagenerator(**test_arguments)

      return train_generator, test_generator


def loop_train (model, optimizer, train_noisy_speech, train_clean_speech, train_batch_losses):

      with tf.GradientTape() as tape:
            train_predict_speech = model(train_noisy_speech)
            # train_loss = weighted_SDR_loss (train_noisy_speech, train_predict_speech, train_clean_speech)
            train_loss = SDR_loss(train_predict_speech, train_clean_speech)
            # train_loss = SDR_loss(train_noisy_speech, train_predict_speech) # 실수

      gradients = tape.gradient(train_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      return train_loss


def loop_test (model, test_noisy_speech, test_clean_speech, test_batch_losses):
      
      'Test loop do not caclultae gradient and backpropagation'
      test_predict_speech = model(test_noisy_speech)
      # test_loss = weighted_SDR_loss (test_noisy_speech, test_predict_speech, test_clean_speech)
      test_loss = SDR_loss(test_predict_speech, test_clean_speech)

      return test_loss


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def model_flow
      def learning_rate_scheduler
            call loop_train
            call loop_test
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def model_flow (model, total_epochs, train_generator, test_generator):

      def learning_rate_scheduler (epoch, learning_rate):

            if (epoch+1) <= 30:
                  return 1.00 * learning_rate
            elif (epoch+1) > 30 and (epoch+1) <= 60:
                  return 0.20 * learning_rate
            else:
                  return 0.05 * learning_rate

      # DEFINE TRAIN STEP, TEST STEP
      train_step = len(os.listdir(train_noisy_path)) // batch_size
      test_step  = len(os.listdir(test_noisy_path)) // batch_size
      print("""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""")
      print("TRAIN STEPS, TEST STEPS   ", train_step, test_step)
      print("""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""")

      for epoch in tqdm(range (total_epochs)):

            train_batch_losses = 0
            test_batch_losses  = 0
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler(epoch, learning_rate), beta_1 = 0.9)

            'Training Loop'
            for index, (train_noisy_speech, train_clean_speech) in tqdm(enumerate(train_generator)):
                  loss = loop_train (model, optimizer, train_noisy_speech, train_clean_speech, train_batch_losses)
                  train_batch_losses = train_batch_losses + loss

            'Test Loop'
            for index, (test_noisy_speech, test_clean_speech) in tqdm(enumerate(test_generator)):
                  loss  = loop_test (model, test_noisy_speech, test_clean_speech, test_batch_losses)
                  test_batch_losses  = test_batch_losses + loss

            'Calculate loss per batch data'
            train_loss = train_batch_losses / train_step
            test_loss  = test_batch_losses / test_step


            templet = 'Epoch  :  {:3d},        TRAIN LOSS  :  {:.6f},        TEST LOSS  :  {:.6f}'
            print(templet.format(epoch+1, train_loss.numpy(), test_loss.numpy()))
            print(optimizer.learning_rate.numpy())

            if ((epoch+1) % 10) == 0: 
                  model.save_weights("./model_save/" + model_type + str(epoch+1) + ".h5")



if __name__ == "__main__":

      parser = argparse.ArgumentParser(description = 'MODEL SETTING OPTION...')
      parser.add_argument("--model", type = str, default = "dcunet20", help = "Input model tpe")
      parser.add_argument("--epoch", type = int, default = 100, help = "Input epochs")
      parser.add_argument("--batch", type = int, default = 64, help = "Input batch size")
      parser.add_argument("--optim", type = str, default = "adam",  help = "Input optimizer option")
      parser.add_argument("--lr",    type = float, default = 0.001, help = "Inputs learning rate")
      parser.add_argument("--trn",   type = str, default = "./datasets/subset_noisy/", help = "Input train noisy path")
      parser.add_argument("--trc",   type = str, default = "./datasets/subset_clean/", help = "Input train clean phat")
      parser.add_argument("--ten",   type = str, default = "./datasets/test_noisy/",  help = "Input test noisy path")
      parser.add_argument("--tec",   type = str, default = "./datasets/test_clean/",  help = "Input test clean path")
      args = parser.parse_args()

      model_type     = args.model
      total_epochs   = args.epoch
      batch_size     = args.batch
      optimizer_type = args.optim
      learning_rate  = args.lr
      train_noisy_path = args.trn
      train_clean_path = args.trc
      test_noisy_path  = args.ten
      test_clean_path  = args.tec


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

if   model_type == "naive_dcunet16":
      selected_model = Naive_DCUnet16().model()
      selected_model.summary()
elif model_type == "naive_dcunet20":
      selected_model = Naive_DCUnet20().model()
      selected_model.summary()
elif model_type == "dcunet16":
      selected_model = DCUnet16().model()
      selected_model.summary()
elif model_type == "dcunet20":
      selected_model = DCUnet20().model()
      selected_model.summary()

model_flow (selected_model, total_epochs, train_generator, test_generator)