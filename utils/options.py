import argparse

def options():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--emergency', action='store_true', 
                        help='False: start ros connection, True: start only yolo')
    argparser.add_argument('-ms', '--multistep', action='store_true', help='mode multistep')
    argparser.add_argument('-d', '--demo', action='store_true', 
                        help='False: start ros connection, True: start only yolo')
    
    argparser.add_argument('-m', '--model', type=str,
                           default ='./saved_models/CollapseNet/collapsenet_model.json', 
                           # default='./trained_data/01_CollapseNet2_batch_32.json',
                           help='Path to model architecture')
    argparser.add_argument('-w', '--weight', type=str,
                           default ='./saved_models/CollapseNet/collapsenet_weight.hdf5', 
                           # default='./trained_data/01_CollapseNet2_batch_32.hdf5',
                           help='Path to weight (trained)')

    argparser.add_argument('-r', '--result', type=str,
                           default='./result',
                           help='Path to image')

    argparser.add_argument('--sample_demo', type=str,
                           default='./sample_demo',
                           help='sample target')

    argparser.add_argument('-i', '--image', type=str,
                           default='./test.jpg',
                           help='Path to image (target)')
    argparser.add_argument('-t', '--test', type=str,
                           default='./test',
                           help='Path to directory (input data)')

    argparser.add_argument('--dataset_name', type=str, default='collapse_data', help='project name')

    argparser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
    argparser.add_argument('--name', type=str, default='collapse_data', help='project name')
    argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
    argparser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    argparser.add_argument('--val_rate', type=float, default=0.1, help='the ratio of validation data')

    opt = argparser.parse_args()
    return opt
