# Transformer-Summarization
 
This model is an Abstractive Summarization model based on [offical Tensorflow models](https://github.com/tensorflow/models). I changed the code to align the peformance with other Transformer based abstractive summarization models (https://github.com/tensorflow/tensor2tensor, https://github.com/nlpyang/PreSumm). You can check out what have changed on below. 

## Requirements
Python3.6

tensorflow==1.14.0 

and install dependencies

    pip3 install --user -r /path/to/models/official/requirements.txt


## Result 
We test the model on CNN/Daily dataset.

| model | Rouge 1  |  Rouge 2  |  Rouge L  |  
| ------ | ------ | ------ | ------ | 
|[Pytorch ver.](https://arxiv.org/pdf/1908.08345.pdf) | 40.21 |17.76 | 37.09 |
| Vanilla Transformer  | 39.01 |17.08 | 36.00|
| Vanilla Transformer + Truncated  | 39.66 |17.52 | 36.60 | 
| Vanilla Transformer + Truncated + Trigram_blocking  | 40.21 |17.78 | 37.16 | 
| Vanilla Transformer + Truncated + Trigram_blocking + CLS_Token  | 40.22 | 17.63 | 37.35 | 

*Vanilla Transformer : Transformer from [offical Tensorflow models](https://github.com/tensorflow/models) with learning rate schedule same as [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)

**Truncated : truncated encode length to 512 and decode length to 128 and skip articles where the length is longer then 2048.

*** Trigram_blocking : Prevent from spitting trigrams more then twice in inference time. 

*** CLS_token : Add CLS token before every sentence. 

## Usage
### 1. Make .tfrecord file  (CNN/Daily dataset)
I used two kinds of tfrecord dataset. 

***Original*** : One is from (http://github.com/tensorflow/tensor2tensor) which has only *inputs* and *targets* features. 

***Expanded*** : The other one is that i've made to add more features such as seperate tokens in BERT (Make sure all those related file names end with 'cls'). So follow the instructions if you want to add more featrues. 

**A. Get the json file** : You can get the sentence splitted dataset as **json format**  by following instructions from [here](https://github.com/nlpyang/PreSumm).

**B. Change paths of ```create_TF.sh``` and ```create_test.sh``` :** Set the *vocab_file* and *data_dir* to your file path. 

**C. run the shell scripts** :
    
    ./create_TF.sh 
    ./create_test.sh
    
you can get .tfrecord file from *create_TF* which will be used for train, evaluation and .txt file from *create_test* which will be used for test.

### 2. Train the model 
All you need to do is just change the path in the script file and run the shell script. 

**A. Train with the *Original* dataset** ```train.sh```

    #set the current path to where the transformer_main.py code exist 
    cd /root/workspace/summarization_tf1.14/models/official/transformer 
    #Ensure that PYTHONPATH is correctly defined as described in
    export PYTHONPATH="/root/workspace/summarization_tf1.14/models"
    #I use 4 gpus change it if you use diff number of gpus
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
        
    # if you want to change model hyper parameters, go to
    # /path/to/models/offical/transformer/model/model_params.py
    
after you change the path run the script file. 
    
    ./train.sh

**B. Train with the *Expanded* dataset**
It goes same with the ```train_cls.sh```. The shell script file runs the ```transformer_main_cls.py```. 

### 3. Model Inference
There are 4 versions of Inference code.
*A. **Original** with Beam search:*  ```test.sh```

*B. **Original** with Beam search + Trigram_blocking:* ```test_triblock.sh```

*C. **Expanded** with Beam search:* ```test_cls.sh```

*D. **Expanded** with Beam search + Trigram_blocking:* ```test_triblock_cls.sh```

( Trigram_blocking :  trigrams are blocked during the beam search where the idea is from the paper [A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION](https://arxiv.org/pdf/1705.04304.pdf).)

The Inference shell script code is as follows,     
    
    #set the current path to where the transformer_main.py code exist 
    cd /root/workspace/summarization_tf1.14/models/official/transformer
    #Ensure that PYTHONPATH is correctly defined as described in
    # https://github.com/tensorflow/models/tree/master/official#requirements
    export PYTHONPATH="/root/workspace/summarization_tf1.14/models"
    # if you want to run the trigram block code, you should run it with the CPU.
    export CUDA_VISIBLE_DEVICES=5
    # Export variables
    PARAM_SET=base
    #Directory where the test article exist.( A text file that each line is an article.)
    DATA_DIR=/root/workspace/summarization_tf1.14/t2t_data
    #Directory where the checkpoints exist. 
    MODEL_DIR=/root/workspace/summarization_tf1.14/models/pretrained_summ
    VOCAB_FILE=$DATA_DIR/vocab.summarize_cnn_dailymail32k.32768.subwords
    #Make sure the train and inference vocabulary file is different 
    DECODE_VOCAB_FILE=$DATA_DIR/vocab_decode.subwords
    
    # Translate some text using the trained model
    #underscored_ids are not used token ids which have underbar in the end (ex: "qqqqq_")
    # Those are used for trigram blocking 
    python translate_triblock.py --model_dir=$MODEL_DIR --vocab_file=$DECODE_VOCAB_FILE --underscored_ids="32371,32369,32366,32361,32353,32349" \
        --param_set=$PARAM_SET --file=$DATA_DIR/test_article_top50.txt  --file_out=$DATA_DIR/test_article_result_top50.txt

after you change the path run the script file. 
   
    ./test_triblock.sh
    
### Visualization




