from glob import glob
from sklearn.model_selection import train_test_split    
import numpy as np                                      # Installed on DAIS 1
from PIL import Image                                   # Installed on DAIS 1
from matplotlib import pyplot as plt                    # Installed on DAIS 1



# Precondition: file name, size of batch to process, and path of output txt file
# Postcondition: saves info to txt file
def create_names_with_batch(filePath, batch_size, save_info_path, quantity):
    array = glob(filePath + '/*.flt')
    validate, test = train_test_split(array[0: quantity], test_size = 0.9)
    t1,t2 = train_test_split(test, test_size = 0.5)
    outFile = open(save_info_path + '/trainning_data_batch%d.txt'%(batch_size),"a")
    out_validate = open(save_info_path + '/validate_data_batch%d.txt'%(batch_size),"a")
    

    outFile.write((str)(batch_size) +'\n')
    out_validate.write((str)(batch_size) +'\n')
    # create validation data
    for k in range(0, len(validate)):
        out_validate.write(validate[k] + '\n')
    out_validate.close()

    #Splits data at proper size
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

# Loads the names used for trainning in the batch formation specified by the txt file generated
# from create_names_with_batch
# Postcondition: array with names of images to be processed
def load_train_text_names(file_path_in):
    outFile = open(file_path_in,"r")
    processing_batch = outFile.readline()
    processing_batch = processing_batch.replace('\n','')
    processing_batch = int(processing_batch)
    arr1 = []
    arr2 = []
    arr1_t = []
    temp = outFile.readline()
    counter = 0
    # Reads until stop
    while temp !='~\n': 
        arr1_t.append(temp.replace('\n',''))
        counter+=1
        temp = outFile.readline()
        if(counter == processing_batch):
            counter = 0
            arr1.append(arr1_t)
            del arr1_t
            arr1_t = []

    temp = outFile.readline()
    # Reads until end
    while temp:
        arr2.append(temp.replace('\n',''))        
        temp = outFile.readline()
    outFile.close()
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    arr2 = arr2.reshape((-1,processing_batch))
    return arr1, arr2

def load_validation_text_names(file_path_in):
    outFile = open(file_path_in,"r")
    processing_batch = outFile.readline()
    processing_batch = processing_batch.replace('\n','')
    processing_batch = int(processing_batch)
    arr = []
    # Start iteration
    temp = outFile.readline()
    while temp:
        arr.append(temp.replace('\n',''))        
        temp = outFile.readline()
    outFile.close()
    arr = np.asarray(arr)
    arr = arr.reshape((-1,processing_batch))
    return arr

# Precondition: file name to print array, and array of longs to be printed
# Postcondition: Image save in file path
def save_image(filePath, image):
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
