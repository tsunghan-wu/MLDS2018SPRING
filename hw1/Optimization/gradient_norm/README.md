# 2-2 Observe gradient norm during training

In most training process, we use gradient descent to let the gradient as small as possible. Thus, in this experiment we want to find out what the gradient norm changes during training.

There are some codes in this directories:
1. code: 
	- `util.py`: useful model library
	- `hw1-2-2.py`: training program (which will record gradient norm, loss and accuracy in `csvdir/*.csv`)
	- `csvdir`:
		- `.csv file`: told before
		- `visualize.py`: read those `.csv` file and generate outcome picture
	- png file: outcome
