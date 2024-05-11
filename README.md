# DICTA
DICTA: DynamIc Content-aware TrAnsformer 


## Requirement

* pytorch >= 1.10.2
* numpy >= 1.13.3
* sklearn
* python 3.9
* transformers


## Usage
Original dataset:
* PMC (pubmed): https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
* DBLPv13 (dblp): https://www.aminer.org/citation


### Running
```sh
# DICTA
python main.py --data_path dblp/pubmed --model transformer/rnn/cnn
```

