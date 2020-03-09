import tensorflow as tf 
import numpy as np 
import pickle

Meta_Graph = './uncased_L-12_H-768_A-12/bert_model.ckpt.meta'
Checkpoint = './uncased_L-12_H-768_A-12/bert_model.ckpt'
saver = tf.train.import_meta_graph(Meta_Graph)
with tf.Session() as sess: 
    saver.restore(sess,Checkpoint)
    g = tf.get_default_graph()
    word_embeddings = g.get_tensor_by_name('bert/embeddings/word_embeddings:0')
    a=word_embeddings.eval()
print(a)
with open("bert_word_embeddings","wb") as f:
    pickle.dump(a,f)
"""

with open("bert_word_embeddings","rb") as f:
    bert_w = pickle.load(f)

tf_var = tf.get_variable("tf_variable",initializer=bert_w)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf_var))

"""