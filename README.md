# DANN-tf2
Tensorflow 2.0 implementation of Domain Adversarial Neural Networks (DANN)

## Environment
Python 3.7
Tensorflow 2.0

## Dataset
Office-31 [Download](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
ImageCLEF-DA [Download](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view)

## Training
```
Office-31

python main.py --gpu (gpu id) --dataset office --source amazon --target webcam --network resnet --logdir (log directory) --datadir (dataset directory) --dw 1e-2
```

```
ImageCLEF-DA

python main.py --gpu (gpu id) --dataset image-clef --source i --target p --network resnet --logdir (log directory) --datadir (dataset directory) --dw 1e-2
```
