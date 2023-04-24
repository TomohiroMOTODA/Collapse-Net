import os
import glob
import numpy as np
from PIL import Image, ImageOps

def remove_file(path):
    for p in glob.glob(path):
        if os.path.isfile(p):
            os.remove(p)

def create_candidate_sample(yolo):
        os.makedirs("./tmp/depth", exist_ok=True)
        os.makedirs("./tmp/mask", exist_ok=True)
        os.makedirs("./tmp/output", exist_ok=True)
        
        path_img = './data/sample/test.jpg'
        try:
            image = Image.open(path_img)
            image_copied = image.copy()
        except:
            print('Open Error! Try again!')
        else:
            
            candidate_list_ = []

            # YOLO v3
            _, clusters = yolo.detect_image_and_return(image) 

            image.save('./data/sample/test_result.jpg', quality=95)
            size = image.size

            remove_file ('./tmp/depth/*.jpg')
            remove_file ('./tmp/mask/*.jpg')
            remove_file ('./tmp/output/*.jpg')

            for i, c in enumerate(clusters):
                top     = c[0] # w-min 
                left    = c[1] # h-min
                bottom  = c[2] # w-max
                right   = c[3] # h-max
                
                black_back_image = np.zeros((size[1], size[0], 3))
                image_cropped = image_copied.crop((left, top, right, bottom))
                pil_image = np.asarray(image_cropped, dtype=np.uint8)
                pil_image_binary = np.where(pil_image<50, 0, 255)
                black_back_image[top:bottom, left:right] = pil_image_binary

                # depth image
                input_file_name = './tmp/depth/img_{}.jpg'.format(i)
                pil_image_flipped = ImageOps.flip(image_copied) 
                pil_image_flipped.save(input_file_name, quality=95)

                # targeted binary image
                pil_target_flipped = Image.fromarray(np.flipud(black_back_image.astype('uint8')))
                pil_target = Image.fromarray(black_back_image.astype('uint8'))
                label_green_file_name = './tmp/mask/img_{}.jpg'.format(i)
                pil_target_flipped.save(label_green_file_name, quality=95)

                # transform tensors
                img_depth  = image_copied.resize((256, 256))
                img_binary = pil_target.resize((256, 256))
                img_depth  = np.array(img_depth)
                img_binary = np.array(img_binary)

                candidate_list_.append([img_depth, img_binary])

            return candidate_list_

def create_candidate(opt, yolo, path, is_tmp=False):
        os.makedirs("./tmp/depth", exist_ok=True)
        os.makedirs("./tmp/mask", exist_ok=True)
        os.makedirs("./tmp/output", exist_ok=True)
        
        try:
            image = Image.open(path)
            image_copied = image.copy()
        except:
            print('Open Error! Try again!')
        else:
            
            candidate_list_ = []

            # YOLO v3
            _, clusters = yolo.detect_image_and_return(image) 
            size = image.size

            remove_file ('./tmp/depth/*.jpg')
            remove_file ('./tmp/mask/*.jpg')
            remove_file ('./tmp/output/*.jpg')

            image.save('./tmp/output/img_classified.jpg', quality=95)

            for i, c in enumerate(clusters):
                top     = c[0] # w-min 
                left    = c[1] # h-min
                bottom  = c[2] # w-max
                right   = c[3] # h-max
                
                black_back_image = np.zeros((size[1], size[0], 3))
                image_cropped = image_copied.crop((left, top, right, bottom))
                pil_image = np.asarray(image_cropped, dtype=np.uint8)
                pil_image_binary = np.where(pil_image<50, 0, 255)
                black_back_image[top:bottom, left:right] = pil_image_binary

                # depth image
                pil_image_flipped = ImageOps.flip(image_copied) 
                
                # targeted binary image
                pil_target_flipped = Image.fromarray(np.flipud(black_back_image.astype('uint8')))
                pil_target = Image.fromarray(black_back_image.astype('uint8'))

                if is_tmp:
                    input_file_name = './tmp/depth/img_{}.jpg'.format(i)
                    pil_image_flipped.save(input_file_name, quality=95)

                    label_green_file_name = './tmp/mask/img_{}.jpg'.format(i)
                    pil_target_flipped.save(label_green_file_name, quality=95)

                # transform tensors
                # img_depth  = image_copied.resize((256, 256))
                # img_binary = pil_target.resize((256, 256))
                img_depth  = pil_image_flipped.resize((256, 256))
                img_binary = pil_target_flipped.resize((256, 256))
                img_depth  = np.array(img_depth)
                img_binary = np.array(img_binary)

                candidate_list_.append([img_depth, img_binary])
            return candidate_list_
