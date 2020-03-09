cd /root/workspace/summarization_tf1.14/models/official/transformer

 #Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
export PYTHONPATH="/root/workspace/summarization_tf1.14/models"
export CUDA_VISIBLE_DEVICES=5
# Export variables
PARAM_SET=base
DATA_DIR=/root/workspace/summarization_tf1.14/t2t_data
MODEL_DIR=/root/workspace/summarization_tf1.14/models/CLS_16_summ_512_128_lr0.15_wrm8k_rmlen2048_base
#MODEL_DIR=/root/workspace/summarization_tf1.14/models/CLS_14_240k 
VOCAB_FILE=$DATA_DIR/vocab.summarize_cnn_dailymail32k.32768.subwords
DECODE_VOCAB_FILE=$DATA_DIR/vocab.txt
# Translate some text using the trained model
python translate_triblock_cls.py --model_dir=$MODEL_DIR --vocab_file=$DECODE_VOCAB_FILE --underscored_ids="32372,32370,32367,32362,32354,32350" \
    --param_set=$PARAM_SET --file=$DATA_DIR/test_article_cls.txt  --file_out=$DATA_DIR/test_article_cls16_240k_result.txt


