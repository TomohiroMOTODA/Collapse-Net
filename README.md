# Collapse-Net

## Usage:

(TBD) Extract the weight models as follows:<br>
```
wget "https://drive.google.com/uc?export=download&id=19aqCNBdecRLP0eeiR5LoaAjCScgH98he" -o saved_models.tar.xz
tar -xf saved_models.tar.xz
```

Train:<br>
```
python train.py
```

Prediction (samples):<br>
```
python prediction.py
```
## Options:


## References:
1. Motoda,T; Petit, D.; Wan, W.; Harada, K; Bimanual Shelf Picking Planner Based on Collapse Prediction. 2021 IEEE 17th International Conference on Automation Science and Engineering (CASE), Lyon, France, 2021, pp. 510-515, https://ieeexplore.ieee.org/document/9551507. <br>
2. Motoda, T.; Petit, D.; Nishi, T.; Nagata, K.; Wan, W.; Harada, K. Shelf Replenishment Based on Object Arrangement Detection and Collapse Prediction for Bimanual Manipulation. Robotics 2022, 11, 104. https://doi.org/10.3390/robotics11050104

* I used a Keras implementation of YOLOv3 by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3). 
