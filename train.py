import utility
import model
import loss 
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import time
from tqdm import tqdm
IMAGE_SHAPE = (200,200,3)
DOWNSCALED_IMG_SHAPE = (100,100,3)
OPTIMIZER = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

OUTPUT_MODELS_FOLDER = "output/models/" 
OUTPUT_FOLDER_IMAGES = "output/images/"
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"]= "100"

def train(imgs_train_path,imgs_test_path,output_models_path,output_images_path,batch_size,epochs,epoch_size = 1000,training_images_to_load = 3000,test_images = 50,training_for_generator_each_batch=2,save_data= True):

    train_x,train_y = utility.get_data(imgs_train_path,IMAGE_SHAPE,training_images_to_load)
    test_x,test_y = utility.get_data(imgs_test_path,IMAGE_SHAPE,test_images,False)

    if save_data:
        utility.save_data_as_pickle(train_x,"dataset/operative_data/train_x")
        utility.save_data_as_pickle(train_y,"dataset/operative_data/train_y")
        utility.save_data_as_pickle(test_x,"dataset/operative_data/test_x")
        utility.save_data_as_pickle(test_y,"dataset/operative_data/test_y")
        print("--SAVED DATA")


    generator = model.get_generator(IMAGE_SHAPE)
    discriminator = model.get_discriminator(IMAGE_SHAPE)
    loss_generator = loss.VGG_LOSS(IMAGE_SHAPE)
    generator.compile(loss=loss_generator.vgg_loss, optimizer=OPTIMIZER)
    discriminator.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    gan = model.get_gan_model(DOWNSCALED_IMG_SHAPE,generator,discriminator,loss_generator.vgg_loss,"binary_crossentropy",OPTIMIZER)
    gan.summary()

    n_batch = int((epoch_size)/ batch_size)
    n_batch_test = int((test_images/batch_size))

    true_batch_vector = np.ones((batch_size,1))
    false_batch_vector = np.zeros((batch_size,1))
    

    print("--START TRAINING")
    for epoch in range(epochs):

        disciminator_losses = []
        gan_losses = []
        epoch_start_time = time.time()

        # train each batch
        for batch in range(n_batch):
            random_indexes = np.random.randint(0, len(train_x), size=batch_size)

            batch_x  =  np.array(train_x)[random_indexes.astype(int)]
            batch_y =  np.array(train_y)[random_indexes.astype(int)]

            generated_images = generator.predict(x=batch_x, batch_size=batch_size)

            discriminator.trainable = True

            #we can decide to perform more than one train for batch on the discriminator 
            for _ in range(training_for_generator_each_batch):
                d_loss_r = discriminator.train_on_batch(batch_y, true_batch_vector)
                d_loss_f = discriminator.train_on_batch(generated_images, np.random.random_sample(batch_size)*0.2)      

                disciminator_losses.append(0.5 * np.add(d_loss_f, d_loss_r))

            discriminator.trainable = False
            
            # train the generator 
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            gan_loss = gan.train_on_batch(batch_x, [batch_y, gan_Y])
            gan_losses.append(gan_loss)

        test_losses = []
        for i in range(n_batch_test):
            batch_x = np.array(test_x)[i*batch_size:(i+1)*batch_size]
            batch_y = np.array(test_y)[i*batch_size:(i+1)*batch_size]
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            gan_loss = gan.test_on_batch(batch_x, [batch_y, gan_Y])
            test_losses.append(gan_loss)

        #print("discriminator loss: ",np.mean(disciminator_losses), " gan losses: ",[np.mean(x) for x in zip(*gan_losses)] ," time: ",time.time()-epoch_start_time)
        print("test: ",[np.mean(x) for x in zip(*test_losses)])

        if epoch % 3 == 0 or epoch == 0:
            generator.save(output_models_path + 'gen_model%d.h5' % epoch)
            discriminator.save(output_models_path + 'dis_model%d.h5' % epoch)
            utility.plot_generated_images(output_images_path,epoch,generator,test_y,test_x)



if __name__ == "__main__":
    train("dataset/val2017/train","dataset/val2017/test",OUTPUT_MODELS_FOLDER,OUTPUT_FOLDER_IMAGES,5,10000,training_for_generator_each_batch = 1)
