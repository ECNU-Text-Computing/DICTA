2023-05-18 20:57:14,284 root         INFO     Namespace(data_path='po', expt_dir='./experiment', log_level='info', model='rnn')
2023-05-18 20:57:14,285 sentence_transformers.SentenceTransformer INFO     Load pretrained SentenceTransformer: all-MiniLM-L6-v2
2023-05-18 20:57:14,698 sentence_transformers.SentenceTransformer INFO     Use pytorch device: cuda
2023-05-18 20:57:16,854 seq2seq.trainer.supervised_trainer INFO     Optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
), Scheduler: None
2023-05-18 20:57:27,260 seq2seq.trainer.supervised_trainer INFO     Progress: 5%, Train RMSE Loss: 0.0704
2023-05-18 20:57:32,088 seq2seq.trainer.supervised_trainer INFO     Progress: 8%, Train RMSE Loss: 0.0288
2023-05-18 20:57:37,861 seq2seq.trainer.supervised_trainer INFO     
Finished epoch 1: Train RMSE Loss: 0.0324, Dev RMSE Loss: 0.1597, Test RMSE Loss: 0.1599
2023-05-18 20:57:41,461 seq2seq.trainer.supervised_trainer INFO     Progress: 11%, Train RMSE Loss: 0.0271
