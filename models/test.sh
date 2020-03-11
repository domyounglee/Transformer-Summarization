cd /root/workspace/summarization_tf1.14/models/official/transformer

 #Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
export PYTHONPATH="/root/workspace/summarization_tf1.14/models"
export CUDA_VISIBLE_DEVICES=0
# Export variables
PARAM_SET=base
DATA_DIR=/root/workspace/summarization_tf1.14/t2t_data
MODEL_DIR=/root/workspace/summarization_tf1.14/models/summ_512_128_lr0.15_t2tlr_wrm8k_rmlen2048_base
VOCAB_FILE=$DATA_DIR/vocab.summarize_cnn_dailymail32k.32768.subwords
DECODE_VOCAB_FILE=$DATA_DIR/vocab_decode.subwords

# Translate some text using the trained model
python translate.py --model_dir=$MODEL_DIR --vocab_file=$DECODE_VOCAB_FILE \
    --param_set=$PARAM_SET --file=$DATA_DIR/test_article00.txt  --file_out=$DATA_DIR/triblock_test_output_encode512_decode128_alpha08_beam5_lr015_rm00.txt



