from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Precondition: file name
# Postcondition: array with names of images to be processed
def load_names(filePath):
    return train_test_split(glob(filePath + '/*.flt'), test_size = 0.5)

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
