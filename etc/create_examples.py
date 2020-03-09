
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import modeling
import tokenization
import tensorflow as tf
import struct
from tensorflow.core.example import example_pb2
import numpy as np 
tf.enable_eager_execution()

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

#flags.DEFINE_string(
#    "output_file", None,
#    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_bool(
    "make_eval", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")




def read_sumarization_examples(input_files):
  """ just tokenize with whitespace """

  Summ_Example = collections.namedtuple(  # pylint: disable=invalid-name
        "Summ_Example", ["article_text", "abstract_text","article_toks", "abstract_toks"])

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples=[]
  for input_file in input_files:

    reader = open(input_file, 'rb')
    print("asdf")

    #for each instance (from : https://github.com/abisee/pointer-generator/blob/master/data.py )
    while True:
      #read the byte 
      len_bytes = reader.read(8)
     
      if not len_bytes: break
      #unpack to value 
      str_len = struct.unpack('q', len_bytes)[0]
      
      example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
      instance = example_pb2.Example.FromString(example_str)
      abstract = str(instance.features.feature['abstract'].bytes_list.value[0].decode())
      article = str(instance.features.feature['article'].bytes_list.value[0].decode())
      
      abstract_toks=[]
      prev_is_whitespace = True
      for c in abstract: 
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            abstract_toks.append(c)
          else:
            abstract_toks[-1] += c
          prev_is_whitespace = False
      
      
      article_toks=[]
      prev_is_whitespace = True
      for c in article: 
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            article_toks.append(c)
          else:
            article_toks[-1] += c
          prev_is_whitespace = False
      
      
      examples.append(Summ_Example(article_text=article, abstract_text=abstract, 
                          article_toks=article_toks, abstract_toks=abstract_toks))
  
  return examples

def create_features(examples, tokenizer,max_art_len,max_abs_len):
  Summ_Feature = collections.namedtuple(  # pylint: disable=invalid-name
        "Summ_Feature", ["unique_id", "art_input_ids","art_segment_ids","art_input_mask", 
        "abs_input_ids","art","abs"])

  #generate a input_ids
  features=[]


  for i,example in enumerate(examples):
    art_toks=example.article_toks
    abs_toks=list(filter(lambda x: x!='<s>' and x!='</s>',example.abstract_toks))


    cleaned_art=" ".join(art_toks)
    cleaned_abs=" ".join(abs_toks)

    art_toks = tokenizer.tokenize(cleaned_art)
    abs_toks = tokenizer.tokenize(cleaned_abs)

    #cuz CLS and SEP
    if len(art_toks)>max_art_len-2:
      art_toks=art_toks[:max_art_len-2]
    if len(abs_toks)>max_abs_len-2:
      abs_toks=abs_toks[:max_abs_len-2]


    art_toks=["[CLS]"]+art_toks+["[SEP]"]
    abs_toks=["[CLS]"]+abs_toks+["[SEP]"]


    art_input_ids = tokenizer.convert_tokens_to_ids(art_toks)
    abs_input_ids = tokenizer.convert_tokens_to_ids(abs_toks)

    
    art_input_mask = [1] * len(art_input_ids)
    art_segment_ids = [0] * max_art_len
    
    while len(art_input_ids) < max_art_len:
      art_input_ids.append(0)
      art_input_mask.append(0)
    while len(abs_input_ids) < max_abs_len:
      abs_input_ids.append(0)

    
    assert len(art_input_ids) == max_art_len
    assert len(art_input_mask) == max_art_len
    assert len(art_segment_ids) == max_art_len
    assert len(abs_input_ids) == max_abs_len


    #print(art_input_ids)
    #print(abs_input_ids)

    features.append(Summ_Feature(unique_id=i,
                                art_input_ids=art_input_ids,
                                art_segment_ids=art_segment_ids, 
                                art_input_mask=art_input_mask, 
                                abs_input_ids=abs_input_ids,
                                art=cleaned_art,
                                abs=cleaned_abs))
  return features
  



def get_enc_output_dec_input(bert_config, is_training, input_ids, input_mask, segment_ids, target_ids, vocab_size,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  enc_output = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(enc_output, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  word_embedding_table = model.get_embedding_table()
  flat_input_ids = tf.reshape(target_ids,[-1])
  one_hot_input = tf.one_hot(flat_input_ids, vocab_size)
  #[batch*seq_len, hiddens_size ]
  dec_input = tf.matmul(one_hot_input, word_embedding_table)

  return enc_output, dec_input





def input_fn_builder(train_features, is_training, repeat, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  """unique_id", "art_input_ids","art_segment_ids","art_input_mask", 
          "abs_input_ids","abs_segment_ids","abs_input_mask"""

  art_input_ids = np.array(
      [f.art_input_ids for f in train_features], dtype='i8')
  art_segment_ids = np.array(
      [f.art_segment_ids for f in train_features], dtype='i8')
  art_input_mask = np.array(
      [f.art_input_mask for f in train_features], dtype='i8')

  abs_input_ids = np.array(
      [f.abs_input_ids for f in train_features], dtype='i8')

  all_unique_ids = np.array(
      [[f.unique_id] for f in train_features], dtype='i8')



  def parse_fn(elem0,elem1,elem2,elem3,elem4):

    SentenceBatch = {"unique_ids": elem0,
                  "input_ids":elem1,
                  "segment_ids":elem2,
                  "input_mask":elem3,
                  "targets":elem4}

    return SentenceBatch



  def input_fn(batch_size):
    """The actual input function."""


    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    
    d = tf.data.Dataset.from_tensor_slices(
    	(all_unique_ids, art_input_ids, art_segment_ids, art_input_mask, abs_input_ids  ))

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            parse_fn,
            batch_size=batch_size,
            drop_remainder=drop_remainder))


    # Prefetch the next element to improve speed of input pipeline.

    d = d.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return d

  return input_fn



def make_eval(features):
  f_art=open("finished_files/article_eval.txt",'w')
  f_abs=open("finished_files/abstract_eval.txt",'w')
  for feature in features:
    f_art.write(feature.art+"\n")
    f_abs.write(feature.abs+"\n")
  f_art.close()
  f_abs.close()

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


  tf.logging.info("*** Reading from input files ***")
  rng = random.Random(FLAGS.random_seed)
  
  #input_file = "/Users/idomyeong/Deeplearning/Summerization/finished_files/chunked/train_287.bin"
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))
  


  examples_ = read_sumarization_examples(input_files)
  features_ = create_features(examples_, tokenizer,512,100)
  if FLAGS.make_eval:
    #make dataset for evaluation 
    make_eval(features_)

  input_f = input_fn_builder(features_,True, 1,True )
  a = input_f(2)
  for i in a:
    print(i)
    break

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  #flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
