# Code for "Saliency Attack: Towards Imperceptible Black-box Adversarial Attack"
This code is for reproducing the results in the paper, [Saliency Attack: Towards Imperceptible Black-box Adversarial Attack](https://dl.acm.org/doi/10.1145/3582563), just accepted by ACM Transactions on Intelligent Systems and Technology.

## Installation
* Python 3.6
* TensorFlow 1.15.0 (with GPU support)
* opencv-python
* Pillow

## Prerequisites
1. Install the required libraries:
```angular2html
pip install -r requirements.txt
```

2. Download ImageNet validation dataset (images and corresponding labels). Note that the validation images must be contained within a folder named `val` and the filename of validation labels must be `val.txt`.
* For images
```
mkdir val
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar -C val
```
* For labels
```
wget http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xvzf caffe_ilsvrc12.tar.gz val.txt
```

3. Place the directory `val` and the file `val.txt` in the same directory.

4. Download a pretrained Inception-v3 model from [Tensorflow model library](https://github.com/tensorflow/models/tree/master/research/slim) and decompress it.
```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
```

5. Set `IMAGENET_PATH` in `main.py` and `MODEL_DIR` in `tools/inception_v3_imagenet.py` to the locations of the dataset and the model respectively.

* For saliency maps
6. We provide 1000 saliency maps in the directory 'saliency-maps' for test. They are generated by
[Pyramid Feature Attention Network for Saliency Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.html), 
as introduced in our paper. We also provide the [implementation](https://github.com/Daizy97/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection) of generating and saving saliency maps with different thresholds.

## How to run
```
python main.py --sample_size 1000 --epsilon 0.05 --max_queries 10000 --block_size 16
```