cd /root/workspace/summarization_tf1.14/models/official/transformer

 #Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
export PYTHONPATH="/root/workspace/summarization_tf1.14/models"
export CUDA_VISIBLE_DEVICES=5
# Export variables
PARAM_SET=base
DATA_DIR=/root/workspace/summarization_tf1.14/t2t_data
MODEL_DIR=/root/workspace/summarization_tf1.14/models/pretrained_summ
VOCAB_FILE=$DATA_DIR/vocab.summarize_cnn_dailymail32k.32768.subwords
DECODE_VOCAB_FILE=$DATA_DIR/vocab_decode.subwords

# Translate some text using the trained model
python translate_triblock.py --model_dir=$MODEL_DIR --vocab_file=$DECODE_VOCAB_FILE --underscored_ids="32371,32369,32366,32361,32353,32349" \
    --param_set=$PARAM_SET --file=$DATA_DIR/test_article_top50.txt  --file_out=$DATA_DIR/test_article_result_top50.txt



