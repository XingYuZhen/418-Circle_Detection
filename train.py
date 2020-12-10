
import tensorflow as tf


ROUND = 10     # Number of training rounds
BATCH_SIZE = 16  #The number of data fed into the neural network at a time
CLASSES = 2  # The number of classes
image_height = 64 
image_width = 64
channels = 1  #Number of channels for training images

  

def get_datasets():
    
    train_data = tf.keras.preprocessing.image.ImageDataGenerator(  #Picture generator
        rescale= 1.0  #Normalization
    )
    train_generator = train_data.flow_from_directory("./data/train/",target_size=(image_height, image_width),color_mode="grayscale",batch_size=BATCH_SIZE,shuffle=True,class_mode="categorical")

# For flow_from_directory The first parameter is the location of the file, the second parameter is the size of the target file, and the third parameter is the color space of the image file
#The fourth parameter is the number of data fed to the neural network at a time, the fifth parameter is the random seed, the sixth parameter is whether to disrupt the order, and the seventh parameter is the method of ImageDataGenerator

    test_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0
    )
    test_generator = test_data.flow_from_directory("./data/validation/", target_size=(image_height, image_width),color_mode="grayscale", batch_size=BATCH_SIZE, shuffle=False,class_mode="categorical" )

    train_number = train_generator.samples
    test_number = test_generator.samples

    return train_generator, test_generator, train_number, test_number


def get_model():
    model = tf.keras.Sequential([
    
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),strides=1,padding="same",activation=tf.keras.activations.relu,input_shape=(64, 64, 1)),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2,padding="same"),
        tf.keras.layers.BatchNormalization(),
       
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=1,padding="same",activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2,padding="same"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=CLASSES, activation=tf.keras.activations.softmax)
    ])

    return model


if __name__ == '__main__':

    train_generator, test_generator, train_number, test_number = get_datasets()

    model = get_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
    model.summary() #Output the parameter status of each layer of the model

    model.fit_generator(train_generator,epochs=ROUND,steps_per_epoch=train_number // BATCH_SIZE,validation_data=test_generator,validation_steps=test_number // BATCH_SIZE)
    #Training model method, save more memory
    model.save("./model/model.h5")
