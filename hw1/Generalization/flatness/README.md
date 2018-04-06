# 3-3 Flatness v.s. Generalization

In this experiment, we are trying to figureout the the ralationship between flatness and generalization. 

In the first part, we will first train two model with same network structure but with defferent training approach like batch and learning rate and then record the loss and accuracy of the linear interpolation between two models.

In the second part, we will train 5 models with different training approach and record the sensitivity of the model.

## Interpolation

There are some files in the directory `interpolation/`:
1. code : 
	- `1-3-3-1.py` : training process (which will record training and testing accuracy + loss in `.npy` file)
	- `visual1-3-3-1.py` : read those `.npy` file and plot the outcome picture
2. `.npy` file : told before
3. png file : the outcome

## Sensitivity

There are some files in the directory `sensitivity/`:
1. code:
	- `1-3-3-2.py` : training process (which will record training and testing accuracy + loss + sensitivity in `.npy` file)
	- `visual-1-3-3-2.py` : read `.npy` file and plot the outcome picture
2. `.npy` file : 
3. png file : the outcome

## Sharpness

There are some files in the directories `bonus/`:
1. code:
	- `1-3-bonus,py`: training process (which will output some usrful data in `.npy` file)
	- `visualize.py`: read `.npy` file and plot the outcome picture
2. `.npy` file : told before
3. png file : the outcome

