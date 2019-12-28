# Detecting semantic anomalies

Code to accompany [Detecting semantic anomalies, AAAI 2020] (https://arxiv.org/abs/1908.04388).

The `rotation` folder contains code for the rotation experiments, and the `cpc` folder for CPC. 
The dataloaders for the Imagenet subsets are based off of a lab-internal format of ILSVRC2012, so you'd have to swap out my Fuel-based dataloader for yours. 

## Requirements and Usage   
Requirements are Python 2.7, TensorFlow v1.13.1 (I've used the wheel on my cluster), Numpy, Scipy, Scikit-Learn, Matplotlib.

I've run most experiments with 4 GPUs.

The commands for running rotation experiments are
```
python main.py -d cifar10 -dc 160 -bs 128 -ngpu 4 -dr 0.3 -c 0 [-r]
python main.py -d stl10 -dc 64 -bs 64 -ngpu 4 -dr 0.3 -c 0 [-r]
```
where `-c` denotes the held out class, and `-r` would augment with rotation-prediction.

Similar usage applies for CPC, with `-s` for adding CPC as a task.
```
python main.py -d dog -c 0 [-s]
```

# Citation
BibTex:
```
@proceedings{ahmed2019semantic,
  title={Detecting semantic anomalies},  
  author={Ahmed, Faruk and Courville, Aaron},  
  booktitle={Proceedings of 34th AAAI Conference on Artificial Intelligence},  
  year={2020}  
}
```
