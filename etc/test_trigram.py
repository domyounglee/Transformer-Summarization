
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.transformer.model import attention_layer
from official.transformer.model import beam_search_triblock as beam_search
from official.transformer.model import embedding_layer
from official.transformer.model import ffn_layer
from official.transformer.model import model_utils
from official.transformer.utils.tokenizer import EOS_ID

_NEG_INF = -1e9


####domyoung 2019.12.9###
from official.transformer.utils import tokenizer

import numpy as np
import random
tf.enable_eager_execution()
def trigram_check(nontri,elem):
	for i in range(len(nontri)-2):
		if (nontri[i],nontri[i+1],nontri[i+2]) == (nontri[-2],nontri[-1],elem):
			return False
	return True
	

def nontrigram_generator(max_len,cand):
	"""
	max_len(int) is the size of the nontrigram array
	cand : list of word_ids that will be used to make nontrigrams (list length should be greater than the 7)
	ex : [32370,32368,32365,32360, 32352, 32351, 32348]
  """

	
	nontri=[]
	while(len(nontri)!=max_len):
		if len(nontri)<3:
			nontri.append(cand[random.randint(0,len(cand)-1)])
			continue
		while(True):
			elem = cand[random.randint(0,len(cand)-1)]		
			if  trigram_check(nontri,elem):
				nontri.append(elem)
				break
	#check
	for i in range(max_len-3):
		for j in range(i+1,max_len-2):
			if (nontri[i],nontri[i+1],nontri[i+2]) == (nontri[j],nontri[j+1],nontri[j+2]):
				return None
	return nontri
########################

max_decode_length=128
nontrigrams= nontrigram_generator(max_decode_length,[32371,32369,32366,32361, 32353, 32352, 32349])
nontrigrams= tf.constant(nontrigrams,dtype=tf.int32)
tile_dims = [1]*nontrigrams.shape.ndims
tile_dims[-1] = 5
nontrigrams = tf.tile(nontrigrams, tile_dims)
nontrigrams = tf.reshape(nontrigrams, [-1,max_decode_length])
print("+"*1000)
print(nontrigrams)
