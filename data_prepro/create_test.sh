export PYTHONPATH="/home/user/dmlee/summarization_tf1.14/models/"
BASE_DIR="/home/user/dmlee"
python3 create_test.py --vocab_file=$BASE_DIR/summarization_tf1.14/t2t_data/vocab.txt --data_dir=$BASE_DIR/data_prepro/more_refined_raw_stories
