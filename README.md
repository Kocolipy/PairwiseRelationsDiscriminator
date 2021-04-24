## Dependencies

The following is a list of dependencies:

    pip install torch
    pip install torchvision
    pip install scikit-image


# Instructions for replication of results

This document serves as an instruction manual for replicating the results found in the paper "Pairwise Relations Discriminator for Unsupervised Raven's Progressive Matrices". 

## 1. Download the RAVEN-10000 dataset [[1]](#1)

We will be using this dataset of [Raven's Progressive Matrices (RPM)](https://drive.google.com/file/d/111swnEzAY2NfZgeyAhVwQujMjRUfeyuY/view) for both training and testing of the model. The [I-RAVEN](https://github.com/husheng12345/SRAN) dataset [[2]](#2) can also be used.

## 2. Unzip the RAVEN-10000 dataset

This should produce a directory with seven subfolders, each for a specific configuration. Each subfolder should contain 10,000 RPM problems. Move the directory into the current working directory (cwd).

## 3. Preprocess the data


To preprocess the data in the RAVEN dataset into a format we can easily use:

``python3 preprocess.py``


The script will produce a data directory as follows:
```
  cwd
  |-- RAVEN-10000
  |-- data
    |-- train
    |-- test
    |-- val
  |-- various python scripts
  ```

  The train directory will contain 42,000 files named 0 to 41,999. The files can be organised into 7 partitions of 6,000 files. Each partition will correspond to the configurations [*Center*, *2x2Grid*, *3x3Grid*, *Out-InCenter*, *Out-InGrid*, *Left-Right*, *Up-Down*] respectively.   i.e. files 6,000 - 11,999 are RPM problems with *2x2Grid* configurations; files 24,000 - 29,999 are RPM problems with *Out-InGrid* configurations.

The test and val directories will each contain 14,000 files named 0 to 1,999. Files are partitioned into blocks of 2,000 and follows the same matching as the partitions in train.

## 4. Train the model

To train the model:

``python3 train.py``

Training will take 200 epochs. The model will take up 4.2 G.B of RAM. Do ensure that your CPU/GPU have sufficient memory.

### Notes: 
Instead of train.py, we can use train_specific.py to train the model using data from only 1 configuration. 

## 5. Evaluate the model

To evaluate the model on the test data:

``python3 evaluate.py --start_ckpt 150 --end_ckpt 200``

It randomly selects 5 checkpoints between start_ckpt and end_ckpt. The average performance of the 5 checkpoints are reported.


# Results

### RAVEN Dataset       
Method | Avg  |  Center  | 2x2Grid | 3x3Grid | L-R | U-D  | O-IC  | O-IG  
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
Random | 12.5 | 12.5 |12.5 |12.5 |12.5 |12.5 |12.5 |12.5 
MCPT [[3]](#3)| 28.5 | 35.9| 26.0| 27.2| 29.3| 27.4| 33.1| 20.7
PRD | **37.9** | **57.8**| **26.8**| **24.8**| **43.4**| **43.3**| **46.7**| **22.9**

### I-RAVEN Dataset    
Method | Avg  |  Center  | 2x2Grid | 3x3Grid | L-R | U-D  | O-IC  | O-IG  
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
Random | 12.5 | 12.5 |12.5 |12.5 |12.5 |12.5 |12.5 |12.5 
PRD | **55.9** | **73.1**| **39.9**| **35.3**| **67.3**| **67.3**| **68.1**| **40.60**

# Pre-trained Model
Pre-trained models for I-RAVEN can be found [here](https://drive.google.com/file/d/1eX5wdmxZg2v29AOUcEsnCxGGxsYEkCM4/view?usp=sharing)

## References
<a id="1">[1]</a> Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Raven: A dataset for relational and analogical visual reasoning, 2019.

<a id="2">[2]</a> Hu, Sheng and Ma, Yuqing and Liu, Xianglong and Wei, Yanlu and Bai, Shihao. Stratified Rule-Aware Network for Abstract Visual Reasoning, 2021.

<a id="3">[3]</a> Tao Zhuo and Mohan Kankanhalli. Solving raven’s progressive matrices with neural networks, 2020.
