bash final4_model/merge_file.sh
python3 video_data.py $1
python3 caption_predict.py $2
