# 1-2 Train on an actual task

In this experiment, we will train on MNIST and CIDAR-10 and see whether deep network structure is better than the shallow one.

## MNIST

In the directory `task_MNIST` are some files below:

1. code:
	- `models.py`: the library of three model
	- `hw1-2.py` : the code on training (which will output `.csv` file)
		- **Note: If you want to reproduce our code, you might need to modified line 15 to change models**
	- `err_summary.py`: read `.csv` file and generate two picture.
2. model: All checkpoints are stored in three directories
3. csv file: All `.csv` file are the output of `hw1-2.py`
4. png file: The outcome

## CIFAR-10

In the directory `task_CIFAR` are some files below:

1. code:
	- `models.py`: the library of three model
	- `hw1-2-2.py` : the code on training (which will output `.csv` file)
		- **Note: If you want to reproduce our code, you might need to modified line 38 to change models**
	- `err_summary.py`: read `.csv` file and generate two picture.
2. model: All checkpoints are stored in three directories
3. csv file: All `.csv` files are the output of `hw1-2-2.py`
4. png file: The outcome

## Result

Deep is better than shallow in this experiment <3
