import os
import glob
import numpy as np
from PIL import Image
from timeit import default_timer as timer

import tensorflow as tf
from keras import backend
from keras.models import model_from_json

from utils.yolo import yolo_for_task
from utils.options import options
from utils.candidates import create_candidate, create_candidate_sample
from utils.visualizer import map_visualizer, map_visualizer_sample

def init_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    backend.set_session(sess)

def init_yolo_detector():
    detector = yolo_for_task()
    return detector

# load the trained model
def load_collapse_net(opt):
    model = model_from_json(open(opt.model).read()) 
    model.load_weights(opt.weight)
    return model

def generate_map_sample(net):
    # load an image
    img_depth  = Image.open("./data/sample/test.jpg")
    img_binary = Image.open("./data/sample/test_mask.jpg")
    img_depth = img_depth.resize((256, 256))
    img_binary = img_binary.resize((256, 256))
    img_depth = np.array(img_depth)
    img_binary = np.array(img_binary)

    # predict by the trained model
    predict = net.predict([[img_depth], [img_binary]])

    map_visualizer_sample(img_depth, img_binary, predict)

def generate_map(opt, net, cand, name='test'):
    if net == None:
        return
    if len(cand) == 0:
        return 

    for i, c in enumerate(cand):
        img_depth  = c[0]
        img_binary = c[1] 
        predict = net.predict([[img_depth], [img_binary]])

        path = os.path.join(opt.result, '%s_%d.jpg'%(name, i)) 
        map_visualizer(opt, c[0], c[1], predict, file_name=path)

if __name__=='__main__':
    init_tensorflow() # initialize the cuDNN
    opt = options() # command options 

    detector = init_yolo_detector()
    net      = load_collapse_net(opt)

    # --- sample
    # cand     = create_candidate_sample(detector)
    # generate_map (opt, net, cand)

    # --- sample_demo (./sample_demo/*.jpg)
    files = glob.glob(os.path.join(opt.sample_demo, '*.jpg'))
    for i, file in enumerate(files):
        start = timer()
        cand = create_candidate(opt, detector, file)
        generate_map(opt, net, cand, name='%d'%(i))
        end = timer()

        print ('computational time: %f' % (end-start))
