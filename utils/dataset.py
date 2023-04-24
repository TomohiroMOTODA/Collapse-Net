import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# definitions
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def addNoise(img, rate = 0.02, seed=1):
    # 乱数によって最大rateを調整して，0(%)~100*seed(%)で変動させる．
    # np.random.seed(seed=seed) # 固定する場合
    param_pepper = np.random.rand()

    # (0.5-a) * 100 % の割合でごま塩
    a = 0.50 - rate * param_pepper
    b = 1.00
    pepper = (b-a) * np.random.rand(img.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1) + a
    pepper = np.round (pepper)

    # RGBの成分を一律にするための工夫
    rand = np.concatenate([pepper, pepper, pepper], 3)
    noisy = img * rand
    return noisy

def get_palette_class():
        palette = [
            [255,   0,   0],
            [  0, 255,   0],
            [  0,   0, 255],
            [  0,   0,   0]
        ]
        return np.asarray(palette)

def adjustData_class_CollapseNet(img, img2, mask):
    img2 = np.where(img2 < 30, 0, img2)
    mask = np.where(mask < 30, 0, mask) 

    if np.max(img) > 1:
        img  = img  / 255
    if np.max(img2) > 1:
        img2 = img2 / 255

    palette = get_palette_class() 

    DEMENSIONS = 2 # ２次元なら赤と黒のみの分類
    onehot = np.zeros((mask.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, DEMENSIONS), dtype=np.uint8) # とりあえず3次元で統一
    for i in range(DEMENSIONS):
        
        if DEMENSIONS == 2: 
            if i == 0:       
                cat_color = [255, 255, 255]
            if i == 1:
                cat_color = [0, 0, 0]
        else:
            cat_color = palette[i]

        temp = np.where((mask[:, :, :, 0] == cat_color[0]) &
                        (mask[:, :, :, 1] == cat_color[1]) &
                        (mask[:, :, :, 2] == cat_color[2]), 1, 0)

        onehot[:, :, :, i] = temp
    return img, img2, onehot

def trainGenerator(image_folder, batch_size=1 ,horizontal_flip=True, save_to_dir=[None, None, None]):
    data_gen_args = dict(
        width_shift_range=  0.2,  # 64,   # 元画像上でのシフト量128にzoom_ratioをかけてint型で設定する
        height_shift_range= 0.2,  # 16,  # 同上
        zoom_range= 0.2,          #[0.9, 1.1],   # 512*512の元画像上で256*256分を等倍で切り出したい
        horizontal_flip=horizontal_flip,
        vertical_flip = False, # 垂直の方向の回転
        rescale=None
    )
    seed = 1      # Shuffle時の時の保証のためにseed値は固定

    image_datagen1 = ImageDataGenerator(**data_gen_args)
    image_datagen2 = ImageDataGenerator(**data_gen_args)

    # Memo class_mode 'categorical' means returning the labels encoded as onehot-vector
    image_generator1 = image_datagen1.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['input'],
        batch_size=batch_size,
        seed=seed, 
        class_mode=None,
        save_to_dir=save_to_dir[0]
        )

    image_generator2 = image_datagen2.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['label_green'],
        batch_size=batch_size,
        seed=seed, 
        class_mode=None,
        save_to_dir=save_to_dir[1]
        )

    image_generator_output = image_datagen2.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['output'],
        batch_size=batch_size,
        seed=seed,
        class_mode=None,
        save_to_dir=save_to_dir[2]
        )

    for (img1, img2, output) in zip(image_generator1, image_generator2, image_generator_output):
        input1, input2, mask = adjustData_class_CollapseNet(img1, img2, output)
    
        input1 = addNoise(input1, rate=0.01, seed=seed) # add random noise 
        input2 = addNoise(input2, rate=0.01, seed=seed) # add random noise 

        yield [input1, input2], mask


def testGenerator(image_folder, batch_size=1):
    data_gen_args = dict(
        # 可視化のため位相をずらす
        width_shift_range=0.,   # 元画像上でのシフト量128にzoom_ratioをかけてint型で設定する
        height_shift_range=0.,  # 同上
        zoom_range=0.,  # 512*512の元画像上で256*256分を等倍で切り出したい
        horizontal_flip=False, # 可視化検証のため．
        vertical_flip=False,
        rescale=None           # リスケールはadjustData()でやる
    )

    seed = 1                    # Shuffle時の時の保証のためにseed値は固定

    image_datagen1 = ImageDataGenerator(**data_gen_args)
    image_datagen2 = ImageDataGenerator(**data_gen_args)

    # Memo class_mode 'categorical' means returning the labels encoded as onehot-vector
    image_generator1 = image_datagen1.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['input'],
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode=None
        )

    image_generator2 = image_datagen2.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['label_green'],
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode=None
        )

    image_generator_output = image_datagen2.flow_from_directory(
        directory=image_folder,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        classes=['output'],
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        class_mode=None
        )

    for (img1, img2, output) in zip(image_generator1, image_generator2, image_generator_output):
        input1, input2, mask = adjustData_class_CollapseNet(img1, img2, output)
        yield [input1, input2], mask #mask_default