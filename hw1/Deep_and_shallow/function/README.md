# 1-1 Simulate a function
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

In this experiment, we will train the machine to fit two functions and see whether deep network structure is better than the shallow one.

## 1. Trail Network Structure

1. nwtwork structure: 1-layer, 3-layer, 5-layer
2. parameter: about 10350 parameters

## 2. $$f(x) = \sum_{n=0}^{10} (\frac{1}{2})^n cos(2^n \pi x)$$

In the directory `function1` are some files below:

1. code:
	- `util.py`: library of model
	- `summary2.py`: training code (which will output `.csv` file)
		- **Note: If you want to reproduce our code, you might need to modified some code segment to change models**
	- `summary.py`: read `.csv` file and generate the training result
	- `err_summary.py`: read `.csv` file and generate the training loss
2. model: All checkpoints are stored in three directories
3. csv file: All `.csv` file are the output of `summary.py`
4. png file: The outcome


## 3. $$f(x) = sin(x) + cos(x^2)$$

In the directory `function2` are some files below:

1. code:
	- `util.py`: library of model
	- `summary2.py`: training code (which will output `.csv` file)
		- **Note: If you want to reproduce our code, you might need to modified some code segment to change models**
	- `summary.py`: read `.csv` file and generate the training result
	- `err_summary.py`: read `.csv` file and generate the training loss
2. model: All checkpoints are stored in three directories
3. csv file: All `.csv` file are the output of `summary.py`
4. png file: The outcome

## 4. Result

Deep is better than shallow in this experiment <3

