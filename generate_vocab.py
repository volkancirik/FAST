import io
import sys
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import transformers
import fasttext
import fasttext.util

tokenizer = transformers.BertTokenizer.from_pretrained(
      "bert-base-multilingual-cased", do_lower_case=False)

def load_vocab(fname,min_freq = 10):
  data = json.load(open(fname,'r'))
  print('len(data)',len(data))

  counts = defaultdict(int)
  pbar = tqdm(data)

  for d in pbar:
    instruction = d['instructions'][0].lower().strip()
    tokenized = instruction.split(' ')
    #tokenized = tokenizer.tokenize(instruction)
    for tok in tokenized:
      counts[tok] += 1
  idx = 3
  w2i = {'<PAD>' : 0,
           '<UNK>' : 1,
           '<EOS>' : 2}
  i2w = {0 : '<PAD>',
         1 : '<UNK>',
         2 : '<EOS>'
  }
  
  for tok in counts:
    if counts[tok] >= min_freq:
      w2i[tok] = idx
      i2w[idx] = tok
      idx += 1
  return w2i, i2w
import io

def load_vectors(fname, vocab):
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  data = {}
  for line in fin:
    tokens = line.rstrip().split(' ')
    if tokens[0] not in vocab:
      continue
    data[tokens[0]] = [float(v) for v in tokens[1:]]
  return data

glove = np.load(sys.argv[3])
print(glove.shape)
w2i, i2w  = load_vocab(sys.argv[1])
d = load_vectors(sys.argv[2],w2i)
print(len(w2i),len(d))

vectors = np.zeros((len(d)+3,300))
vectors[0,:] = glove[0,:]
vectors[1,:] = glove[1,:]
vectors[2,:] = glove[2,:]
vocab = ['<PAD>','<UNK>','<EOS>']
idx = 3
for w in d:
  vectors[idx,:] = d[w]
  vocab.append(w)
  idx += 1
lang = sys.argv[4]
np.save('train_glove.{}.npy'.format(lang), vectors)
f = open('train_vocab.{}.txt'.format(lang),'w')
for w in vocab:
  f.write(w+'\n')
f.close()
