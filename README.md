@[TOC](EnClaSC)

# Requirements

- python3(is preferable)
- keras
- tensorflow
- hickle
- lightgbm
- sklearn

# Usage instructions
## Download

Download EnClaSC.
>git clone https://github.com/xy-chen16/EnClaSC

## Data

There are nine datasets in the __Data__ folder including __Baron Dataset, Xin Dataset, Muraro Dataset, Segerstolpe Dataset, Macosco Dataset, Shekhar Dataset, Darmanis Dataset, Romanov Dataset, Zeisel Dataset.__ Each dataset contains three files: XXX_logcounts.csv, XXX_label.csv, XXX_feature.csv.
## Feature Selection
We provide four methods of dimensionality reduction in __data_generate.ipynb__. Modify the path to use it. The following is the correspondence between the name and method：
- PCA——PCA
- myself——EnClaSC
- 460147——Seurat v3.0
- scmap——scmap

## Cell-type classification

Modify the path to use __run_all.py__. 
>run_all.py 0 1 0.3 1
>
The program needs four parameters, the first is the GPU number, the second is the number of dataset , the third is the bagging fration(0.3 in EnClaSC), and the fourth is the number of data selection method.


