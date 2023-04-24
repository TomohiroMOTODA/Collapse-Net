# -*- coding: utf-8 -*-
'''
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install -y graphviz
$ pip install pydot
'''
import os
import glob
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras import callbacks
from keras import backend
import tensorflow as tf

from network.collapse_net import CollapseNet
from utils.options import options
from utils.dataset import trainGenerator, testGenerator

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)

if __name__ == "__main__":
    opt = options()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    backend.set_session(sess)
    
    os.makedirs('./saved_models/CollapseNet/%s' % opt.name, exist_ok=True)

    # learning rate
    def step_decay_editable(epoch):
        lr = opt.lr
        if epoch >= 20:
            lr = lr/10.
        if epoch >= 40:
            lr = lr/100. 
        return lr
    lr_decay = callbacks.LearningRateScheduler(step_decay_editable)

    # set the model
    model = CollapseNet()
    model.summary() # summary
    plot_model(model, 'CollapsNet-architecture.png')

    # folder path
    train_path = 'data/%s/train' % (opt.dataset_name)
    valid_path = './data/%s/valid' % (opt.dataset_name)

    # count dataset
    lines = os.listdir(os.path.join(train_path, 'output'))
    v_lines = os.listdir(os.path.join(valid_path, 'output'))
    num_val = len(v_lines)
    num_train = len(lines)
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train//opt.batch_size, num_val//opt.batch_size, opt.batch_size))

    # dataloader
    train_gen = trainGenerator(train_path, batch_size=opt.batch_size)
    validation_gen = trainGenerator(valid_path, batch_size=opt.batch_size)

    # callback
    os.makedirs('./tmp/checkpoints', exist_ok=True)
    cp_path = './tmp/checkpoints/model_{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.h5' 
    callbacks_list = [
        # save the checkpoints
        callbacks.ModelCheckpoint(filepath=cp_path, monitor='val_loss', save_best_only=True),
        # decay learning rate
        lr_decay,
        # early stopping
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')
    ]

    # save the model
    json_string = model.to_json()
    open(os.path.join('./saved_models/CollapseNet/%s' % opt.name, 'model.json'), 'w').write(json_string)

    # train
    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=num_train//opt.batch_size,
        validation_data=validation_gen,
        validation_steps=num_val//opt.batch_size,
        epochs=opt.n_epoch,
        callbacks=callbacks_list,
        use_multiprocessing=True
    )

    # save the weight
    model.save_weights(os.path.join('saved_models/CollapseNet/%s' % opt.name, 'weight.hdf5'))


    # plot the graph ------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    # -------------------------------------------------------------
