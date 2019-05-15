import os,glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
'''
    RETURN A  LIST OF FILE IN A FOLDER
'''
def get_files_in_folder(path):
    return [ os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) ]

'''
    RETURN AN IMAGE TO THE CORRECT COLOR SPACE
'''
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

'''
    NORMALIZE AN IMAGE MAKIN EWACH COLOR VALUE BE BETWEEN -1,1
'''
def normalize(input_data):
    return(np.asarray(input_data).astype(np.float32) - 127.5)/127.5 

''' 
    LOAD ALL IMAGES IN A FOLDER WITH THE GIVEN SHAPE
'''
def load_images(path,shape,n_images = -1,shuffle = True):
    files = get_files_in_folder(path)
    print(len(files))
    images = [ f for f in files if (f.lower().endswith('.jpg'))]
    if shuffle:
        random.shuffle(images)
    if n_images > len(images):
        raise("Not enought valid images in the dataset folder")
    elif n_images > 0: 
        images = images[:n_images]
    return load_images_of_shape(images,shape)
'''
    LOAD A LIST OF IMAGES RETURNING ONLY THOSE WHOSE SHAPE IS EQUIALS TO THE GIVEN ONE
'''
def load_images_of_shape(images,shape):
    np_images = []
    for img_path in images:
        np_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # Load the image as a numpy array (using the RGB color space)
        if np_image.shape == shape:
            np_images.append(normalize(np_image))
        else:
            print(np_image.shape)
    print(len(np_images))
    return np_images

'''
    ADD ARTIFACTS TO EACH IMAGE
'''
def add_artefacts(images):
    processed_images = []
    for image in images:
        #img = add_uniform_blur(image) 
        img = resize_image(image,(100,100))
        processed_images.append(img)
    return processed_images

'''
    ADD MOTION BLUR IN A RANDOM DIRECTION RANDOMIZINF THE BLUR SIZE BETWEEN THE GIVEN BOUNDARIES
    min_blur and max_blur must be odd
'''
def add_motion_blur(img,min_blur,max_blur):
    size = random.randrange(min_blur, max_blur, 2)
    if size % 2 !=1:
        size = size +1
    kernel_motion_blur = get_random_blur_kernel(size) / size
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output

'''
    GENERATE A RANDOM KERNEL FOR THE ADD_MOTION_BLUR FUNCTION
'''
def get_random_blur_kernel(size):
    if size % 2 !=1:
        raise("size deve essere dispari") 
    pivot = int((size+1)/2)+(1*random_sing)
    kernel = np.zeros((size,size))
    kernel[pivot,pivot] = 1
    shift = random.randint(-1,1)
    for i in range(pivot):
        kernel[pivot-i-1,pivot+shift] = 1
        kernel[pivot+i+1,pivot-shift] = 1
        if shift == 0:
            shift = random.randint(-1,1)
        elif shift<0:
            shift = shift -random.randint(0,1)
        else:
            shift = shift +random.randint(0,1)
    return kernel

'''
    RETURNS 2 LISTS WITH ALL OR A SUBSET OF THE IMAGES IN THE GIVEN PATH
    THE FIRST WILL CONTAINS ALL THE IMAGES THAT THE GENERATOR SHALL RECIVE AS INPUT
    THE SECOND WILL CONTAINS THE CORRESPONDING IMAGES THAT THE GENERATOR WILL HAVE TO COME UP WITH 
'''
def get_data(path,images_shape,n_images = -1,shuffle = True):
    images = load_images(path,images_shape,n_images,shuffle)
    images_with_artifcats = add_artefacts(images)
    return np.array(images_with_artifcats),np.array(images)
'''
    RESIZE AN IMAGE
'''
def resize_image(image,size):
    return cv2.resize(image, size) 
'''
    PLOT IMAGES 
    reference: https://github.com/deepak112/Keras-SRGAN/blob/master/Utils.py
'''
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr , dim=(1, 3), figsize=(15, 5)):
    value = 0
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(x_test_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(x_test_lr)
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
    
    plt.close('all')

def add_uniform_blur(img):
    i = random.randint(3,15)
    if i % 2 == 0:
        i = i + 1 * random_sing()
    return cv2.GaussianBlur(img,(i,i),0)

def random_sing():
    return 1 if random.random() < 0.5 else -1

'''
    SAVE SOME DATA AS PICKLE
'''
def save_data_as_pickle(data,path):
    pickle_file = open(os.path.join(path+".obj"),"wb")
    pickle.dump(data,pickle_file)

def save_image(image,path):
    
    cv2.imwrite(path, image)
    cv2.imshow("image", img)