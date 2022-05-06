# SMAN

Code for Joint Entity-Relation Extraction Task.

Train model
python main.py train --config configs/example_train.conf for conll04 dataset
python main.py train --config configs/example_train_sci.conf for SciERC dataset
python main.py train --config configs/example_train_ade.conf for ADE dataset

Eval model
python main.py eval --config configs/example_eval.conf for conll04 test set
python main.py eval --config configs/example_eval_sci.conf for SciERC test set
python main.py eval --config configs/example_eval_ade.conf for ADE test set

Get our saved optimal model for https://xxxxxxxx (waiting for update).

This repository is based on the excellent open-source projects https://github.com/lavis-nlp/spert. If it is helpful to you, please cite their paper: 
Markus Eberts, Adrian Ulges. Span-based Joint Entity and Relation Extraction with Transformer Pre-training. 24th European Conference on Artificial Intelligence, 2020.
