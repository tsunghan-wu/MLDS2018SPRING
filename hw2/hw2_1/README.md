## MLDS HW2-1 Seq2seq Video Caption

### Package usage

- python standard library (sys, pickle, argparse ...)
- numpy 1.14.2
- pandas 0.22.0
- tensorflow 1.6.0
- nltk 3.2.5

### How to use it

1. Inference mode

	`./hw2_seq2seq.sh $1 $2`

	- $1 is the input folder containing testing video. 
	- $2 is the output file path predicting the caption.

2. Training mode

	- in `model.config`, change the `model_save_path` to the folder you save the model.
	- in `video_data_v2.py`, change the first three path `label_path`,`feat_path`,`test_test` to the corresponding folder and file.
	- use `python3 video_data_v2.py` to create `proccessed_training_data_v2`
	- use `python3 caption.py` to train the model.

### File in this directory

1. Model (execute two shell scripts and you will download it from my dropbox)
	- `final4_model/` : tensorflow model (checkpoint)

2. Python code
	- `video_data.py` : testing data loader / processing source code
	- `video_data_v2.py` : training data loader / processing source code
	- `model.py` : tensorflow seq2seq model
	- `caption.py` : training mode main function
	- `caption_predict.py` : inference mode main function
	
3. Shell script
	- `hw2_seq2seq.sh` : inference mode shell script
	- `final4_model/split_file.sh` : split the model into 50M pieces
	- `final4_model/merge_file.sh` : merge the model pieces into one model file 

4. Report
	- `report.pdf` : which is the report of this experiment

