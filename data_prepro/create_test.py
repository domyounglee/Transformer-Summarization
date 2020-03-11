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




def main(unused_argv):

    subtokenizer = tokenizer.Subtokenizer(FLAGS.vocab_file)



    test_set = glob.glob(FLAGS.data_dir+"/"+"*.test.*.json")
    total_set = {"test":test_set}
    
    print(total_set)
    for mode,_set in total_set.items():
        writer = open(FLAGS.data_dir+"/"+"test_article_cls.txt","w")
        writer2 = open(FLAGS.data_dir+"/"+"test_article_cls_answer.txt","w")
        mode_count = 0
        
        for file_name in _set:
            with open(file_name) as f:
                lines=f.readlines()
                
                for line in lines:
                    instances= json.loads(line)
                    for inst_index,instance in enumerate(instances):
                        
                        for i,src in enumerate(instance['src']):
                          
                          writer.write("CLS "+" ".join(src)+" ")
                        writer.write("\n")

                        for i,tgt in enumerate(instance['tgt']):
                          
                          writer2.write(" "+" ".join(tgt)+" . ")
                        writer2.write("\n")

                        mode_count+=1
    print(mode+" : "+str(mode_count))
    writer.close()
    writer2.close()                  


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
