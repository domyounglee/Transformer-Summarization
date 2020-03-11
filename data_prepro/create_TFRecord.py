import json 
import collections
import glob
from official.transformer.utils import tokenizer_cls as tokenizer
# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order
from official.utils.flags import core as flags_core


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return [tokenizer.CLS_ID] + subtokenizer.encode(line) 

def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  return subtokenizer.decode(ids)
def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def main(unused_argv):

    subtokenizer = tokenizer.Subtokenizer(FLAGS.vocab_file)
    print(subtokenizer.subtoken_list)

    train_set = glob.glob(FLAGS.data_dir+"/"+"*.train.*.json")
    valid_set = glob.glob(FLAGS.data_dir+"/"+"*.valid.*.json")
    test_set = glob.glob(FLAGS.data_dir+"/"+"*.test.*.json")
    total_set = {"train":train_set,"valid":valid_set,"test":test_set}
    
    #print(total_set)
    print_switch=True
    for mode,_set in total_set.items():
        writer = tf.python_io.TFRecordWriter(FLAGS.data_dir+"/"+mode+".tfrecord")
        mode_count = 0
        
        for file_name in _set:
            with open(file_name) as f:
                lines=f.readlines()
                for line in lines:
                    instances= json.loads(line)
                    for inst_index,instance in enumerate(instances):
                        

                        src_list=[]
                        src_sep_list=[]
                        src_cls_mask=[]
                        
                        
                        for i,src in enumerate(instance['src']):
                            src_line=" ".join(src)
                            src_line= src_line.replace("-RRB-",")")
                            src_line= src_line.replace("-LRB-","(")
                            src_line= src_line.replace("-RSB-","]")
                            src_line= src_line.replace("-LSB-","[")
                            src_line= src_line.replace("-RCB-","}")
                            src_line= src_line.replace("-LCB-","{")
                            src_ids = _encode_and_add_eos(src_line,subtokenizer)

                            src_list.extend(src_ids)
                            src_sep_list.extend([i]*len(src_ids))
                            src_cls_mask.extend([1]+[0]*(len(src_ids)-1))


                        src_list+= [tokenizer.EOS_ID]
                        src_sep_list+=[i+1]
                        src_cls_mask+=[0]

                        tgt_list = []
                        tgt_sep_list = []
                        tgt_cls_mask = []
                        

                        for i,tgt in enumerate(instance['tgt']):
                            tgt+=["."] #It doesn't have punctuation 
                            tgt_line=" ".join(tgt)
                            tgt_line= tgt_line.replace("-RRB-",")")
                            tgt_line= tgt_line.replace("-LRB-","(")
                            tgt_line= tgt_line.replace("-RSB-","]")
                            tgt_line= tgt_line.replace("-LSB-","[")
                            tgt_line= tgt_line.replace("-RCB-","}")
                            tgt_line= tgt_line.replace("-LCB-","{")

                            tgt_ids = _encode_and_add_eos(tgt_line,subtokenizer)

                            tgt_list.extend(tgt_ids)       
                            tgt_sep_list.extend([i]*len(tgt_ids))
                            tgt_cls_mask.extend([1]+[0]*(len(tgt_ids)-1))

                            
                        tgt_list+= [tokenizer.EOS_ID]
                        tgt_sep_list+=[i+1]
                        tgt_cls_mask += [0]


                        features = collections.OrderedDict()
                        features["inputs"] = create_int_feature(src_list)
                        features["targets"] = create_int_feature(tgt_list)
                        features["input_seps"] = create_int_feature(src_sep_list)
                        features["target_seps"] = create_int_feature(tgt_sep_list)
                        features["input_cls_mask"] = create_int_feature(src_cls_mask)
                        features["target_cls_mask"] = create_int_feature(tgt_cls_mask)

                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

                        writer.write(tf_example.SerializeToString())

                        #print
                        
                        if inst_index < 10 and print_switch:
                            tf.logging.info("*** Example ***")
                            tf.logging.info("*** INPUT ***")
                            tf.logging.info(_trim_and_decode(src_list,subtokenizer))
                            tf.logging.info("*** TARGET ***")
                            tf.logging.info(_trim_and_decode(tgt_list,subtokenizer))
                            for feature_name in features.keys():
                                feature = features[feature_name]
                                values = []
                                if feature.int64_list.value:
                                    values = feature.int64_list.value
                                elif feature.float_list.value:
                                    values = feature.float_list.value
                                tf.logging.info("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

                        mode_count+=1
                    print_switch=False

        print(mode+" : "+str(mode_count))
        writer.close()
                        


def define_translate_flags():
  """Define flags used for translation script."""
  # Model flags
  flags.DEFINE_string(
      name="data_dir", short_name="md", default="/tmp/transformer_model",
      help=flags_core.help_wrap(
          "Directory containing dataset."))
  flags.DEFINE_string(
      name="vocab_file", short_name="vf", default=None,
      help=flags_core.help_wrap(
          "Path to subtoken vocabulary file. If data_download.py was used to "
          "download and encode the training data, look in the data_dir to find "
          "the vocab file."))
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("data_dir")


if __name__ == "__main__":
  define_translate_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
