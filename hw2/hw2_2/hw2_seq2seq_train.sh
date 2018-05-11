wget https://www.dropbox.com/s/secd4l8k1b96osz/data_class?dl=1 -O data_class
python3 train.py --data_loader data_class --model $1 --log $2
