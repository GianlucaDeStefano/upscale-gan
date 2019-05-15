from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import add


'''
    Add to the model one convolutional layer with nomalization and  activation 
    (Con2d, BatchNormalization, LeakyRelu)
'''
def add_standard_convolutional_layer(model,filters,kernel_size,strides,leaky_relu_alpha = 0.3):
    model = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(model)
    model = BatchNormalization()(model)
    model = LeakyReLU(leaky_relu_alpha)(model)
    return model


'''
    residual block, 
    this block is composed by 2 Convolutional layer (with batchnorm and activation (in this case a Prelu)) 
    then we link directly the output of the block with the input using a  
    residual net theory -> https://en.wikipedia.org/wiki/Residual_neural_network
    credits, implementation and some more explenantions here -> https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64
'''
def add_residual_block(model,filters,kernel_size,strides,_project_shortcut=False):
    initial_model = model

    model = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(model)
    model = BatchNormalization()(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)


    model = Conv2D(filters, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization()(model)

    model = add([initial_model, model])

    return model

'''
    INCREASE THE SIZE OF THE MODELS USING upscaling_factor  
'''
def add_upscaling_block_2D(model,upscaling_factor):
    model = UpSampling2D(size = upscaling_factor)(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    return model

'''
    RETURNS THE GENERATOR MODEL
'''
def get_generator(image_shape):

    gen_input = Input(shape = [None,None,image_shape[2]])
	    
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)       


    for index in range(3):
        model = add_residual_block(model,64,3,1)

    #model = add_upscaling_block_2D(model,2)
    model = add_upscaling_block_2D(model,2)

    #model = add_upscaling_block_2D(model,(2,2))

    #for index in range(1):
    #    model = add_residual_block(model,64,3,1)
    

    model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
    generator_model = Model(inputs = gen_input, outputs = model)
    generator_model.summary()

    return generator_model

'''
    RETURN THE DISCRIMINATOR MODEL  
'''
def get_discriminator(image_shape):     

    dis_input = Input(shape = image_shape)
    
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
    model = LeakyReLU(alpha = 0.2)(model)

    for i in range(10):
        model = add_standard_convolutional_layer(model,64*2^int((i+1)/2), 3, i % 2 + 1) 
    
    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    model = Dense(1)(model)
    model = Activation('sigmoid')(model) 
    
    discriminator_model = Model(inputs = dis_input, outputs = model)

    discriminator_model.summary()
    return discriminator_model
    

'''
    RETURN THE GAN OBJECT AKA A MODEL CONTAINING BOTH GENERATOR AND DISCRIMINATOR 
'''
def get_gan_model(image_shape, generator, discriminator,generator_loss_function,discriminator_loss_function,optimizer):
    discriminator.trainable = False
    inputs = Input(shape=image_shape)
    gen= generator(inputs)
    discriminator = discriminator(gen)
    model = Model(inputs=inputs, outputs=[gen,discriminator]) #[gen,discriminator] -> this way we can support multiple images
    model.compile(loss=[generator_loss_function, discriminator_loss_function],loss_weights=[1., 1e-3], optimizer=optimizer)
    return model



