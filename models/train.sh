cd /root/workspace/summarization_tf1.14/models/official/transformer
 #Ensure that PYTHONPATH is correctly defined as described in
export PYTHONPATH="/root/workspace/summarization_tf1.14/models"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Export variables
PARAM_SET=base
DATA_DIR=/root/workspace/summarization_tf1.14/t2t_data
MODEL_DIR=/root/workspace/summarization_tf1.14/models/tf2_summ_512_128_lr0.2_t2tlr_wrm8k_rmlen2048_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.summarize_cnn_dailymail32k.32768.subwords
DECODE_VOCAB_FILE=$DATA_DIR/vocab_decode.subwords

python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=200000 \
    --steps_between_evals=10000 \
    --batch_size=16384 \
    --num_gpus=4 
    #--hooks=loggingmetrichook \
    #--bleu_source=$DATA_DIR/small_article_eval.txt --bleu_ref=$DATA_DIR/small_abstract_eval.txt


