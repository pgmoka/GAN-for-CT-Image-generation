from utils import *         # File with miscelaneous helper methods
import argparse             # For getting user input        # Installed on DAIS 1

# OS performance:

#Keras imports:                                             # Installed on DAIS 1
import keras                    
from keras.layers import Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Concatenate
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf

# Object created to create, save, load, and test the model
class project:
    def __init__(self, bs, test = 1, load_file = '.', validate_file_in = '.', validate_file_out = '.'):
        if test == 1:
            print("\n-----> Initializing project! <-----\n")
            self.batch_size = bs
            self.shape = (512, 512, 1)
            self.discriminator_filter_number = 64
            self.gerator_filter_number = 64
            
            self.disc_patch = (32, 32, 1)       # Properly adjusted
            # Load data:

            # optimizer_Adam = 'adam'
            # Create discriminator:
            self.discriminator = self.create_discriminator()
            self.discriminator.compile(loss = 'mse',
                        optimizer = Adam(0.001,0.5),
                        metrics = ['accuracy'])
            # Create generator:
            self.generator = self.create_generator()

            img_A = keras.Input(shape = self.shape)
            img_B = keras.Input(shape = self.shape)

            fake = self.generator(img_B)
            
            # Set network for generator:
            self.discriminator.trainable = False
            
            score = self.discriminator([fake, img_B])
            self.combined = Model(inputs = [img_A, img_B], outputs = [score, fake])
            # Change optimizer here:
            self.combined.compile(loss = ['mse', 'mae'], loss_weights=[1,100],optimizer = Adam(0.001,0.5))

            print("-----> End of Model Creation! <-----\n")
            print(self.generator.optimizer)
        else:
            self.generator = keras.models.load_model(load_file)
            self.validate_generator(validate_file_in, validate_file_out, 0)

        

# Postcondition: Creates a generator model, which gets an image input, and an image output
    def create_generator(self):
        print("---> Creating generator!\n")
        # Input
        noise = keras.Input(shape = self.shape)

        # Layers
        #Down
        layer1 = Conv2D(self.gerator_filter_number, kernel_size = 4, strides = 2, padding = 'same')(noise)
        layer1 = LeakyReLU(alpha=0.2)(layer1)

        layer2 = Conv2D(self.gerator_filter_number*2, kernel_size = 4, strides = 2, padding = 'same')(layer1)
        layer2 = LeakyReLU(alpha=0.2)(layer2)
        layer2 = BatchNormalization(momentum=0.8)(layer2)

        layer3 = Conv2D(self.gerator_filter_number*4, kernel_size = 4, strides = 2, padding = 'same')(layer2)
        layer3 = LeakyReLU(alpha=0.2)(layer3)
        layer3 = BatchNormalization(momentum=0.8)(layer3)

        #Up
        layer4 = UpSampling2D(size=2)(layer3)
        layer4 = Conv2D(self.gerator_filter_number*4, kernel_size = 4, strides = 1, padding = 'same', activation='relu')(layer4)
        layer4 = BatchNormalization(momentum=0.8)(layer4)
        layer4 = Concatenate()([layer4, layer2])

        layer5 = UpSampling2D(size=2)(layer4)
        layer5 = Conv2D(self.gerator_filter_number*2, kernel_size = 4, strides = 1, padding = 'same', activation='relu')(layer5)
        layer5 = BatchNormalization(momentum=0.8)(layer5)
        layer5 = Concatenate()([layer5, layer1])

        last_layer = UpSampling2D(size=2)(layer5)
        
        # output:
        image = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(last_layer)
        # SeparableConv2D
        # Layers
        return Model(noise, image)

# Postcondition: model that creates discriminator, with two image inputs and one output
    def create_discriminator(self):
        print("---> Creating discriminator!\n")
        # Inputs:
        img_A = keras.Input(shape = self.shape)
        img_B = keras.Input(shape = self.shape)
        combined = Concatenate(axis=-1)([img_A, img_B])

        # combined = keras.backend.reshape(combined, shape = (self.batch_size,512,512,1))
        # img_A = keras.backend.reshape(img_A, shape = (self.batch_size,512,512,1))
        # img_B = keras.backend.reshape(img_B, shape = (self.batch_size,512,512,1))
        # Layers:
        layer1 = Conv2D(self.discriminator_filter_number, kernel_size = 4, strides = 2, padding = 'same')(combined)
        layer1 = LeakyReLU(alpha=0.2)(layer1)

        layer2 = Conv2D(self.discriminator_filter_number*2, kernel_size = 4, strides = 2, padding = 'same')(layer1)
        layer2 = LeakyReLU(alpha=0.2)(layer2)
        layer2 = BatchNormalization(momentum=0.8)(layer2)

        layer3 = Conv2D(self.discriminator_filter_number*4, kernel_size = 4, strides = 2, padding = 'same')(layer2)
        layer3 = LeakyReLU(alpha=0.2)(layer3)
        layer3 = BatchNormalization(momentum=0.8)(layer3)


        layer4 = Conv2D(self.discriminator_filter_number*8, kernel_size = 4, strides = 2, padding = 'same')(layer3)
        layer4 = LeakyReLU(alpha=0.2)(layer4)
        layer4 = BatchNormalization(momentum=0.8)(layer4)
        # output:
        score = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(layer4)

        return Model([img_A, img_B], score)
    

# Trains model
# Precondition: path file for testing, validation data path, file to save sample, 
# file to save results, number of epochs for trainning, interval for saving information
# Postcondition: trains models in the
    def train_model(self, data_path, validate_path, sample_file, result_file, epoch_number, info_interval):
        print("-----> Trainning model! <-----\n")
        self.train1, self.train2 = load_train_text_names(data_path)
        
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for e in range(0, epoch_number) :
            counter = 0
            avFile = open(result_file + "/report_iteration.txt","a")
            print('\nEpoch: %d\n' % (e))
            for (t1,t2) in zip(self.train1, self.train2):
                # Ground truth for trainning
                
                # #Get data from image, and generate fake
                # real_image1 = load_float(t1)
                # real_image2 = load_float(t2)

                real_image1 = []
                real_image2 = []

                for i in range(0,len(t1)):
                    real_image1.append(load_float(t1[i]))
                    real_image2.append(load_float(t2[i]))

                real_image1 = np.asarray(real_image1)
                real_image2 = np.asarray(real_image2)

                fake_image = self.generator.predict(real_image1)

                # train discriminator
                # Trains on correct images
                loss_real = self.discriminator.train_on_batch([real_image1, real_image2], valid)
                # Trains on fake images
                loss_fake = self.discriminator.train_on_batch([real_image1, fake_image], fake)
                
                total_loss = 0.5 * np.add(fake_image,loss_real)

                # Train generator
                combined_loss = self.combined.train_on_batch([real_image1, real_image2], [valid, real_image1])      # key point
                
                avFile.write("\n------------------ Epoch%d (Batch %d)\n---Discriminator:\n- Loss for real images: %.4f" % (e, counter, loss_real[0]))
                avFile.write("\n- Loss for fake images %.4f\n---Generator:\n- Loss: %.4f" % (loss_fake[0], combined_loss[0]))
                # print("\n------------------ Epoch%d (Batch %d)\n---Discriminator:\n- Loss for real images: %.4f" % (e, counter, loss_real[0]))
                # print("\n- Loss for fake images %.4f\n---Generator:\n- Loss: %.4f" % (loss_fake[0], combined_loss[0]))
                counter +=1
            if e % info_interval == 0 or e == epoch_number:
                print('Image printed at batch: ',e)
                self.validate_generator(validate_path, sample_file,e)
                # save_image(sample_file + '/output_it%d' % (e), fake_image[0])
        avFile.close()

# Precondition: path where location of the path is, location path to output sample, epoch at location
# Postcondition: Images saved
    def validate_generator(self, text_name_path, output_path, epoch):
        test = load_validation_text_names(text_name_path)
        
        batch_counter = 0
        # rotate through different batches
        for v in test:
            img = []
            # transforms words on batch to images
            for i in range(0,len(v)):
                img.append(load_float(v[i]))
            img = np.asarray(img)
            validation_output_data = self.generator.predict(img)

            #prints image
            for j in range(0,len(validation_output_data)):
                isolate = v[j].split('/')[-1].split('\\')[-1].split('.')[0]
                save_image(output_path + '/' + isolate+ '_validate_e%d_b%d_img%d' % (epoch,batch_counter,j), validation_output_data[j])
            batch_counter +=1

# Saves model
#Precondition: string destination to print model at
    def save_model(self, destination):
        print("-----> Saving model! <-----\n")
        self.generator.save(destination + "/generator.model")
        self.discriminator.save(destination + "/discriminator.model")

# called by the command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', dest='in_file', default='./names_log/trainning_data_batch3.txt', help='input file -- directory or single file')
    parser.add_argument('--in_validate', dest='in_val', default='./names_log/validate_data_batch3.txt', help='input file -- directory or single file')    
    parser.add_argument('--sample', dest='sample_file', default='./sample', help='sample file directory for sample images to be saved')
    parser.add_argument('--results', dest='result_file', default='./results', help='result file directory for text related output')
    parser.add_argument('--epochs', dest='epoch_number', default='10', help='number of epochs to train the neural network(int)')
    parser.add_argument('--save_file', dest='save_file', default='./model', help='file to save/load model')
    parser.add_argument('--batch_size', dest='batch_size', default='3', help='number of images per batch to train the neural network')
    parser.add_argument('--info_interval', dest='intev', default='1', help='Intervals where information is saved')
    

    args = parser.parse_args()
    user_in = args.in_file
    batch = args.batch_size


    p = project(int(args.batch_size))
    p.train_model(user_in, args.in_val, args.sample_file, args.result_file, int(args.epoch_number), int(args.intev))
    p.save_model(args.save_file)
    # test = './names_log/trainning_data_batch3.txt'.split('/')[-1].split('.')[0]
    print("\nPROGRAM OVER\n")