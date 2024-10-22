# cellannotation
Cell type annotation

# Environment
```
pip install scanpy
pip install pytorch-ignite
pip install typing_extensions
pip install tensorboardX
pip install torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl
```

# Dataset
Two separate single-cell atlases (demo datasets): https://pan.baidu.com/s/1voWMkeedRUKNoND5gRkF7Q?pwd=xgu1

For other dataset sources, please refer to our manuscript.

# Use
Usage is as follows:

- data: Store reference datasets and query datasets
- notebook_demo: Demonstration of the annotation method on PBMC 10k and two separate single-cell atlases
- results: Result output data and visualizations

Enter the notebook_demo directory and execute the following files in sequence:
```
python 0-data_read.py # View reference dataset and query dataset
python 1-prestep.py # Preparation before training
python 2-train.py # Training on reference dataset
python 3-predict.py # Inference on query dataset
```
