# Manifold-Aware Local Feature Modeling for Semi-Supervised Medical Image Segmentation
Pytorch implementation for "Manifold-Aware Local Feature Modeling for Semi-Supervised Medical Image Segmentation"

## Environment Installation
1. Requirements 
```
Python 3.11.3
CUDA 12.1
PyTorch 2.0.1
```
2. Install Independencies
```
# build conda environment
conda create -n MANet python=3.11.3 -y
conda activate MANet

# install latest PyTorch prebuilt with the default prebuilt CUDA version
conda install pytorch torchvision -c pytorch

# install other dependencies
pip install -r requirements.txt
```
## Usage
#### 1. Download Dataset
You can download three open datasets: LA, ACDC, Panreas-NIH from the Internet.
#### 2. Link dataset path under `$MANet/data`
#### 3. Data Structure
Final data structure is shown like this:
```
MANet/
|---data/
| |---LA/
| | |---2018LA_Seg_Training_Set/
| | |---train.list
| | |---test.list
| |---ACDC/
| | |---data/
| | |---train.list
| | |---test.list
| | |---all_slices.list
| | |---train_slices.list
| | |---val.list
| |---pancreas/
|---BCP/
|---CAML/
|---MCF/
```
#### 3. Train
Run the following command to start a training process. 
```
cd MANet

# e.g., for 10% labels on LA using BCP as baseline model
python BCP/LA_BCP_manifold.py --pretrain_model {Network} --model {Network} --alpha 0.05
```
#### 4. Test
Run the following command to start a testing process.
```
cd MANet

# e.g., for 10% labels on LA using BCP as baseline model
python BCP/test_LA.py --pretrain_model {Network} --model {Network} --alpha 0.05
```
