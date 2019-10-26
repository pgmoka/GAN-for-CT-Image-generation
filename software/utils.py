from glob import glob
from sklearn.model_selection import train_test_split    
import numpy as np                                      # Installed on DAIS 1
from PIL import Image                                   # Installed on DAIS 1
from matplotlib import pyplot as plt                    # Installed on DAIS 1

# Precondition: file name
# Postcondition: array with names of images to be processed
def load_names(filePath):
    return train_test_split(glob(filePath + '/*.flt'), test_size = 0.5)

# Precondition: file name
# Postcondition: array with names of images to be processed
def load_names_with_batch(filePath, batch_size):
    t1,t2 = train_test_split(glob(filePath + '/*.flt'), test_size = 0.5)
    iterator_counter = 0
    x1 = []
    xt1=[]
    x2 = []
    xt2=[]
    for i in range(0, len(t1)):
        if iterator_counter == batch_size:
            x1.append(xt1)
            del xt1
            xt1 = []

            x2.append(xt2)
            del xt2
            xt2 = []

            iterator_counter = 0
        if iterator_counter < batch_size:
            xt1.append(t1[i])
            xt2.append(t2[i])
            iterator_counter += 1

        
    return x1,x2

# Precondition: file name to print array, and array of longs to be printed
# Postcondition: Image save in file path
def save_image(filePath, image):
    # image = np.reshape(image_in , [512, 512])
    # max_pixel = np.amax(image)
    # img = (image/max_pixel) * 255
    # img = np.round(img)
    # plt.figure(num=None, figsize=(30, 40), facecolor='w', edgecolor='k')
    # plt.style.use('grayscale')
    # plt.imshow(img, interpolation = 'nearest')
    # plt.savefig(image)
    # plt.show()
    # plt.close()
    scalef = np.amax(image)
    print_img = np.clip(255 * image/scalef, 0, 255).astype('uint8')
    print_img = np.squeeze(print_img)
    im = Image.fromarray(print_img.astype('uint8')).convert('L')
    im.save(filePath, 'png')

# Precondition: name of file of image
# Postcondition: array based on image
def load_float(fileName):
    float_arr= np.fromfile(fileName, dtype= '<f')
    return float_arr.reshape(1, 512, 512, 1)
