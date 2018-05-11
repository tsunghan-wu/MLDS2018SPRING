bash final4_model/merge_file.sh
python3.5 video_data.py $1
python3.5 caption_predict.py $2
