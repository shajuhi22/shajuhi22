This repository provides an implementation of the research paper "Complex Query Answering with Neural Link Predictors," leveraging Hugging Face API for model comparisons. The goal is to answer complex queries on

# knowledge graphs using neural link predictors while comparing the proposed method with existing models.

* Table of Contents *
  
## Introduction

Knowledge graphs store information as entities and relationships, but they often contain missing data. This project aims to answer complex queries involving logical conjunctions (AND), disjunctions (OR), and existential quantifiers using neural link predictors.

The approach involves translating queries into differentiable objectives and solving them using gradient-based and combinatorial search methods. We compare our framework with state-of-the-art models like Query2Box and Graph Query Embedding using Hugging Face models.

## Dataset

The datasets used in this implementation are:

* 1 FB15k (Freebase knowledge graph) 

* 2 FB15k-237 (Filtered Freebase)

* 3 NELL-995 (Never-Ending Language Learning dataset)

You can download these datasets from Hugging Face Datasets.

## Methodology

* We implement the following approaches:

* Continuous Query Decomposition (CQD-CO): Optimizes query embeddings using gradient descent.

* Beam Search (CQD-Beam): Greedy combinatorial optimization to find optimal assignments.

## Comparison models

* Hugging Face Implementation of Query2Box.

* Hugging Face Implementation of Graph Query Embedding.

## Installation

* Create a conda environment with PyTorch, Cython, and scikit-learn:

'conda create --name kbc_env python=3.7'
'source activate kbc_env'
'conda install --file requirements.txt -c pytorch'

Then install the KBC package to this environment:

python setup.py install

Datasets

To download the datasets, go to the kbc/scripts folder and run:

chmod +x download_data.sh
./download_data.sh

Once the datasets are downloaded, add them to the package data folder by running:

python kbc/process_datasets.py

This will create the files required to compute the filtered metrics.

Usage

Reproduce the results with the following command:

python kbc/learn.py --dataset FB15K --model ComplEx --rank 500 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 1e-2 --max_epochs 100 --valid 5

Commands for running different datasets:

python kbc/learn.py --dataset WN18 --model ComplEx --rank 2000 --optimizer Adagrad --learning_rate 1e-2 --batch_size 100 --regularizer N3 --reg 5e-2 --max_epochs 20

python kbc/learn.py --dataset FB15K-237 --model ComplEx --rank 1000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 5e-2 --max_epochs 100

python kbc/learn.py --dataset YAGO3-10 --model ComplEx --rank 500 --optimizer Adagrad --learning_rate 5e-3 --batch_size 500 --regularizer N3 --reg 5e-3 --max_epochs 50

Example for Hugging Face model comparison:

from transformers import pipeline

query_pipeline = pipeline("question-answering", model="huggingface/query2box")

query = {"question": "Which drugs interact with proteins associated with diseases?", "context": "Knowledge graph data"}

result = query_pipeline(query)
print(result)

Results

Evaluation metrics include:

Hits@3

Mean Reciprocal Rank (MRR)

Execution time comparison

Results comparison:

Feel free to submit pull requests or open issues to improve the repositoryCQD-CO
Use the kbc.cqd_co script to answer queries, providing the path to the dataset, and the saved link predictor trained in the previous step. For example,

% python -m kbc.cqd_co data/FB15k --model_path models/[model_filename].pt --chain_type 1_2


## Final Results
All results from the paper can be produced as follows:

> % cd results/topk
> % ../topk-parse.py *.json | grep rank=1000
> d=FB15K rank=1000 & 0.779 & 0.584 & 0.796 & 0.837 & 0.377 & 0.658 & 0.839 & 0.355
d=FB237 rank=1000 & 0.279 & 0.219 & 0.352 & 0.457 & 0.129 & 0.249 & 0.284 & 0.128
d=NELL rank=1000 & 0.343 & 0.297 & 0.410 & 0.529 & 0.168 & 0.283 & 0.536 & 0.157
% cd ../cont
% ../cont-parse.py *.json | grep rank=1000
d=FB15k rank=1000 & 0.454 & 0.191 & 0.796 & 0.837 & 0.336 & 0.513 & 0.816 & 0.319
d=FB15k-237 rank=1000 & 0.213 & 0.131 & 0.352 & 0.457 & 0.146 & 0.222 & 0.281 & 0.132
d=NELL rank=1000 & 0.265 & 0.220 & 0.410 & 0.529 & 0.196 & 0.302 & 0.531 & 0.194
Generating explanations
When using CQD-Beam for query answering, we can inspect intermediate decisions. We provide an example implementation for the case of 2p queries over FB15k-237, that generates a log file. To generate this log, add the --explain flag when running the cqd_beam script. The file will be saved as explain.log.
