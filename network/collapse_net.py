from keras.models import Model
from keras.layers import Input, MaxPooling2D, Concatenate, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16

# todo
def init_weight():
    return 

def CollapseNet():
    input1 = Input((256, 256, 3))
    input2 = Input((256, 256, 3))

    base_model = VGG16(weights='imagenet',
                       include_top=False, input_tensor=input1)

    # fix some layers (totally 18 layers), if you fix all layers, set [:19] on loop.)
    for layer in base_model.layers[:15]:
        layer.trainable = False

    x = base_model.output

    # ------------------------------------------------------
    y1 = Conv2D(16, (3, 3), padding='same')(input2)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)

    y1 = Conv2D(32, (3, 3), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)

    y1 = Conv2D(32, (3, 3), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)

    y1 = Conv2D(64, (3, 3), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)

    y1 = Conv2D(128, (3, 3), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y1)

    # Concatenation ----------------------------------------
    combined = Concatenate()([x, y1])

    # Skip Connection --------------------------------------
    pool1_x = base_model.get_layer("block1_pool").output
    pool2_x = base_model.get_layer("block2_pool").output
    pool3_x = base_model.get_layer("block3_pool").output
    pool4_x = base_model.get_layer("block4_pool").output


    x = Conv2D(512, (3, 3), padding='same')(combined)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Concatenate()([pool4_x, x]) # 結合

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)

    x = Concatenate()([pool3_x, x]) # 結合

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)

    x = Concatenate()([pool2_x, x]) # 結合

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)

    x = Concatenate()([pool1_x, x]) # 結合

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)

    # Terminal Layer --------------------------------------------------
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(0.2)(x)

    # last convolutional layer -----------------------------------------
    output = Conv2D(filters=2, # 0-back ground, 1-collapse zone, 2-picking targetの３つで良いのかしら...
                    kernel_size=1, 
                    activation='softmax',
                    name='output')(x)

    # Compile step ----------------------------------------------------
    model = Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  )

    return model