import argparse
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # avoid conflict
import cv2
import open3d
import numpy as np

'''
TODO:
NaNの読み込みの問題から，PLY形式に非対応
PCDなら読める．
'''

def options():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, default='./tmp/testRG.py', help='number of epochs')
    argparser.add_argument('--img_height', type=int, default=1024, help='image height') # 256
    argparser.add_argument('--img_width',  type=int, default=1280, help='image width') # 320

    argparser.add_argument('--max_distance',  type=int, default=800, help='image width')
    argparser.add_argument('--min_distance',  type=int, default=600, help='image width')

    opt = argparser.parse_args()
    return opt

if __name__=='__main__':
    opt = options()
    
    pcd = open3d.io.read_point_cloud("./tmp/testRG.pcd", format='pcd', remove_nan_points=False, remove_infinite_points=False, print_progress=True)
    print ('%d points loaded. '%(len(pcd.points)))

    img = np.full((opt.img_height, opt.img_width, 3), 0, dtype=np.uint8)
    count = 0
    
    for y in range (opt.img_height):
        for x in range (opt.img_width):
            z = pcd.points[count][2]
            if z > -999999:
                max_d = float(opt.max_distance)
                min_d = float(opt.min_distance)
                pixel = 1.-(z -min_d)/(max_d-min_d)
                pixel = int(pixel*255)
                
                img[y][x] = [pixel, pixel, pixel] 
                # print (z, pixel)

                if z >= opt.max_distance:
                    img[y][x] = [0, 0, 0]
                if z <= opt.min_distance:
                    img[y][x] = [0, 0, 0]

            else:
                img[y][x] = [0, 0, 0]

            count += 1

    dst = cv2.GaussianBlur(img, (1, 1), 3)
    cv2.imwrite('./tmp/depth.jpg', img)
    
