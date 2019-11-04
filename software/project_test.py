from utils import *         # File with miscelaneous helper methods
import argparse             # For getting user input
import tensorflow as tf     # for creating neural networks

#Keras imports:
import keras
from keras.layers import Concatenate
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Model



class project:
    def __init__(self, in_file, checkpoint_file, sample_file, result_file, epoch):
        print("\n-----> Initializing project! <-----\n")
        self.batch = 20
        self.shape = (512, 512, 1)
        # Load data:
        self.train, self.test = load_names(in_file)

        # Create discriminator:
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(optimizer = 'adam',
                    loss = 'mse',
                    metrics = ['accuracy'])
        # Create generator:
        self.gerator = self.create_generator()

        img_A = keras.Input(shape = self.shape)
        img_B = keras.Input(shape = self.shape)

        # fake = self.gerator(img_B)
        # self.combined = Model(inputs = [img_A, img_B])

        train()
        print("-----> End of Model Creation! <-----\n")
            

    def create_generator(self):
        print("---> Creating generator!\n")
        img_A = keras.Input(shape = self.shape)
        img_B = keras.Input(shape = self.shape)
        combined = Concatenate(axis=-1)([img_A, img_B])

        model = keras.models.Sequential()
        model.add(Flatten())
        model.add(Dense(128,activation=tf.nn.relu))
        model.add(Dense(128,activation=tf.nn.relu))
        model.add(Dense(10,activation=tf.nn.softmax))
        return model

        # mnits = tf.keras.datasets.mnist
        # (x_train, y_train), (x_test, y_test) = mnits.load_data()
        # x_train = tf.keras.utils.normalize(x_train, axis = 1)

        # x_test = tf.keras.utils.normalize(x_test, axis = 1)

        # # layer1 = tf.keras.Input(shape = (512,512,1))
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
        # model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
        # model.compile(optimizer = 'adam',
        #             loss = 'sparse_categorical_crossentropy',
        #             metrics = ['accuracy'])
        # model.fit(x_train, y_train, epochs = 3)
        # # model.save('test.model')
        # # new_model = tf.keras.models.load_model('test.model')
        # # predictions = new_model.predict([x_test])
        # # print(np.argmax(predictions[0]))

    def create_discriminator(self):
        print("---> Creating discriminator!\n")
        img_A = keras.Input(shape = self.shape)
        img_B = keras.Input(shape = self.shape)
        combined = Concatenate(axis=-1)([img_A, img_B])

        model = keras.models.Sequential()
        model.add(Flatten())
        model.add(Dense(128,activation=tf.nn.relu))
        model.add(Dense(128,activation=tf.nn.relu))
        model.add(Dense(10,activation=tf.nn.softmax))
        return model
    

# Trains model
def train():
    print("---> Trainning model!\n")

# Saves model
def save_model(modelToBeSaved):
    print("---> Saving model!\n")

# called by the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', dest='in_file', default='./data', help='input file -- directory or single file')
    parser.add_argument('--checkpoint', dest='checkpoint_file', default='./checkpoints', help='checkpoint file for checkpoints to be saved')
    parser.add_argument('--sample', dest='sample_file', default='./sample', help='sample file directory for sample images to be saved')
    parser.add_argument('--results', dest='result_file', default='./results', help='result file directory for text related output')
    parser.add_argument('--epochs', dest='epoch_number', default=200, help='number of epochs to train the neural network')
    args = parser.parse_args()
    p = project(args.in_file, args.checkpoint_file, args.sample_file, args.result_file, args.epoch_number)