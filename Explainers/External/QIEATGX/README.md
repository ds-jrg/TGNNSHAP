# QIEA-TGX

QIEA-TGX is a post-hoc explainer for temporal graph neural networks, leveraging a quantum-inspired evolutionary algorithm. This work has been accepted to AAAI 2026.


# How to run

## Download wikipedia and reddit datasets
Download from http://snap.stanford.edu/jodie/wikipedia.csv and http://snap.stanford.edu/jodie/reddit.csv and put them into ~/workspace/dataset/data


## Create environment
Move to ~/workspace and create your environment by Dockerfile 

## Preprocess real-world datasets
```
cd  ~/workspace/TGNNmodels/xgraph/models/ext/tgat
python process.py -d wikipedia
python process.py -d reddit
```


## Generate explain indexs
```
cd  ~/workspace/dataset
python tg_dataset.py -d wikipedia(or reddit) -c index
```

## Train tgat/tgn model
tgat:
```
cd  ~/workspace/tgnnexplainer/xgraph/models/ext/tgat
./train.sh
./cpckpt.sh
```

tgn:
```
cd  ~/workspace/tgnnexplainer/xgraph/models/ext/tgn
./train.sh
./cpckpt.sh
```

## Create target events and get neighbors
We remain target events using in tha paper.

if you need new target events:
```
cd  ~/workspace/dataset/test_data
./create_targets.sh
``` 

else:   skip this section and use neigbors in ~/workspace/dataset/hops

## Run our explainer and other  baselines
```
cd  ~/workspace/Xmethods/codes
./run_sp.sh
``` 

# Reference

```bibtex
@inproceedings{mitani2026explainer,
  title        = {Explaining Temporal Graph Neural Networks via Quantum-Inspired Evolutionary Algorithm},
  author       = {Mitani, Masahiro and Sasaki, Yuya},
  booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year         = {2026}
}
```