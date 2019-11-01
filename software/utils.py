from glob import glob
from sklearn.model_selection import train_test_split    
import numpy as np                                      # Installed on DAIS 1
from PIL import Image                                   # Installed on DAIS 1
from matplotlib import pyplot as plt                    # Installed on DAIS 1

# Precondition: file name
# Postcondition: array with names of images to be processed
def create_names(filePath):
    files1, files2 = train_test_split(glob(filePath + '/*.flt'), test_size = 0.5)
    return files1, files2

# Precondition: file name
# Postcondition: array with names of images to be processed
def create_names_with_batch(filePath, batch_size, save_info_path):
    t1,t2 = train_test_split(glob(filePath + '/*.flt'), test_size = 0.5)
    # iterator_counter = 0
    # x1 = []
    # xt1=[]
    # x2 = []
    # xt2=[]
    outFile = open(save_info_path,"a")
    
    outFile.write((str)(batch_size) +'\n')

    # for i in range(0, len(t1)):
    #     if iterator_counter == batch_size:
    #         x1.append(xt1)
    #         del xt1
    #         xt1 = []

    #         x2.append(xt2)
    #         del xt2
    #         xt2 = []

    #         iterator_counter = 0
    #     if iterator_counter < batch_size:
    #         xt1.append(t1[i])
    #         xt2.append(t2[i])
    #         iterator_counter += 1
    # x1 = np.asarray(x1)
    # x2 = np.asarray(x2)

    true_size = len(t1)
    if(len(t1)>len(t2)):
        true_size = len(t2)
    true_size = true_size - (true_size%batch_size)
    # Saves info to outfile
    for i in range(0, true_size):
        outFile.write(t1[i] + '\n')
    #Midpoint stop
    outFile.write('~\n')
    
    for j in range(0,true_size):
        outFile.write(t2[j] + '\n')
    outFile.close()
    # return x1,x2

def load_text_names(file_path_in):
    outFile = open(file_path_in,"r")
    processing_batch = outFile.readline()
    processing_batch = processing_batch.replace('\n','')
    processing_batch = int(processing_batch)
    arr1 = []
    arr2 = []

    temp = outFile.readline()
    while temp !='~\n':
        arr1.append(temp)
        temp = outFile.readline()
    while temp:
        arr2.append(temp)        
        temp = outFile.readline()
    outFile.close()
    arr1 = np.asarray(arr1)
    arr1 = arr1.reshape((-1,processing_batch))
    arr2 = np.asarray(arr2)
    arr2 = arr2.reshape((-1,processing_batch))
    return arr1, arr2

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
    image.tofile(filePath + '.flt')

    scalef = np.amax(image)
    print_img = np.clip(255 * image/scalef, 0, 255).astype('uint8')
    print_img = np.squeeze(print_img)
    im = Image.fromarray(print_img.astype('uint8')).convert('L')
    im.save(filePath+'.png', 'png')

# Precondition: name of file of image
# Postcondition: array based on image
def load_float(fileName):
    float_arr= np.fromfile(fileName, dtype= '<f')
    return float_arr.reshape(512, 512, 1)
