# DICTA
DICTA: DynamIc Content-aware TrAnsformer 


## Requirement

* pytorch >= 1.10.2
* numpy >= 1.13.3
* sentence_transformers
* python 3.9
* transformers


## Usage
Original dataset:
* PMC (pubmed): https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
* DBLPv13 (dblp): https://www.aminer.org/citation

Download the dealt dataset from https://drive.google.com/file/d/1VOSjqvnKu04MAnA6dVrhpdzm9WH4msKS/view?usp=sharing


### Running
```sh
# DICTA
python main.py --data_path dblp/pubmed --model transformer/rnn/cnn
```

