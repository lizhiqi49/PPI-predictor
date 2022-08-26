# PPI-predictor
A graph neural network (GNN) model to predict protein-protein interactions (PPI) with no sample features



### Overview

This is a simple graph convolutional network (GCN) to predict the protein-protein interactions. There are 2 datasets: a large one and a small one. The useful information in the dataset are only known protein-protein interactions and the bioinformatic database query of proteins. Since crawling more information from the public database is troublesome, in this project, I want to predict PPIs with only their known interaction relationships, so GCN is utilized.



### Software requirements

* Python 3.8.3
* PyTorch 1.6.0
* torch_geometric



### Repo content explanation

* *dataset* folder contains two .txt files: a larger dataset and a small dataset.
* `train.py`  is the script defining the GCN model and training it.
* `metrics.py`: compute the metrics for performance evaluation.
* `topInteract.py`: choose the protein-protein pairs with highest score as the predicted PPIs.
* *\*_files* folders contain the output files when training based on different dataset.





