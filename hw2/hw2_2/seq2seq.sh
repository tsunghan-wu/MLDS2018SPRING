wget https://www.dropbox.com/s/secd4l8k1b96osz/data_class?dl=1 -O data_class
wget https://www.dropbox.com/s/52wunvg4tv3o508/final_model.tar.gz?dl=1 -O final_model.tar.gz
tar zxvf final_model.tar.gz
rm final_model.tar.gz
python3 reload.py --data_loader data_class --model ./final_model/checkpoint.ckpt --test $1 --out $2
