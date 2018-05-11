## MLDS HW2-2 Seq2seq Chatbot

### Package usage

- python standard library (sys, pickle, argparse ...)
- numpy 1.14.2
- pandas 0.22.0
- tensorflow 1.16.0

### How to use it

1. Inference mode

	`bash hw2_seq2seq.sh $1 $2`

	- $1 is the input file preprocessed by `jieba` (use space to split chinese words)
	- $2 is the output file

2. Training mode

	`bash hw2_seq2seq_train.sh $1 $2`

	- $1 is the path where you want to save your model
	- $2 is log file path

### File in this directory

1. Model (execute two shell scripts and you will download it from my dropbox)
	- `data_class` : data loader / processing pickle file
	- `model/` : tensorflow model (checkpoint)

2. Python code
	- `data_v5` : data loader / processing source code
	- `model.py` : tensorflow seq2seq model
	- `model_seq2seq.py` : training mode main function
	- `reload.py` : inference mode main function

3. Shell script
	- `hw2_seq2seq.sh` : inference mode shell script
	- `hw2_seq2seq_train.sh` : training mode shell script

4. Report
	- `report.pdf` : which is the report of this experiment
