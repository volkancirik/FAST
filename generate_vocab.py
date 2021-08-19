usage = '''
    python generate_vocab.py <train file> <fasttext vec file> <glove npy of the repo> <language code> <0,1 for r2r or refer360images> <wordvec name e.g. glove>
    python generate_vocab.py data/RxR_en-ALL_train.json data/cc.en.300.vec data/trainval_glove.npy en-ALL 0 glove
    python generate_vocab.py /projects/vcirik/refer360/data/continuous_grounding/train.[all].imdb.npy data/cc.en.300.vec data/trainval_glove.npy refer360 1 glove
'''
import io
import sys
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import transformers
import fasttext
import fasttext.util
import re
tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", do_lower_case=False)


def load_vocab(fname, min_freq=10):
  data = json.load(open(fname, 'r'))
  print('len(data)', len(data))

  counts = defaultdict(int)
  pbar = tqdm(data)

  for d in pbar:
    instruction = d['instructions'][0].lower().strip()
    tokenized = instruction.split(' ')

    #tokenized = tokenizer.tokenize(instruction)
    for tok in tokenized:
      token = tok.split('.')[0]
      token = tok.split(',')[0]
      token = tok.split('\'')[0]
      if len(token):
        counts[token] += 1
  idx = 3
  w2i = {'<PAD>': 0,
         '<UNK>': 1,
         '<EOS>': 2}
  i2w = {0: '<PAD>',
         1: '<UNK>',
         2: '<EOS>'
         }

  for tok in counts:
    if counts[tok] >= min_freq:
      w2i[tok] = idx
      i2w[idx] = tok
      idx += 1
  return w2i, i2w


def load_refer360_vocab(fname, min_freq=1):

  dump = np.load(fname, allow_pickle=True)[()]
  data = dump['sentences']

  print('len(data)', len(data))

  counts = defaultdict(int)
  pbar = tqdm(data)

  SENTENCE_SPLIT_REGEX = re.compile(
      r'(\W+)')  # Split on any non-alphanumeric character

  for tokenized in pbar:

    instruction = ' '.join(tokenized)
    tokenized_regex = [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(
        instruction.strip()) if len(s.strip()) > 0]

    for tok in tokenized_regex:
      #token = re.split(',. |_;', tok)[0]
      token = tok
      if len(token):
        counts[token] += 1

    for tok in tokenized:
      counts[tok] += 1
  idx = 3
  w2i = {'<PAD>': 0,
         '<UNK>': 1,
         '<EOS>': 2}
  i2w = {0: '<PAD>',
         1: '<UNK>',
         2: '<EOS>'
         }
# unk ratio: 0.03 15339.0 598081.0
  for tok in counts:
    if counts[tok] >= min_freq:
      w2i[tok] = idx
      i2w[idx] = tok
      idx += 1
  return w2i, i2w


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


if __name__ == '__main__':
  wordvec = np.load(sys.argv[3])
  print(wordvec.shape)

  if sys.argv[5] == '0':
    LV = load_vocab
  elif sys.argv[5] == '1':
    LV = load_refer360_vocab
  else:
    raise NotImplementedError()
  w2i, i2w = LV(sys.argv[1])
  d = load_vectors(sys.argv[2], w2i)
  print('w2i & d', len(w2i), len(d))

  vectors = np.zeros((len(d)+3, 300))
  vectors[0, :] = wordvec[0, :]
  vectors[1, :] = wordvec[1, :]
  vectors[2, :] = wordvec[2, :]
  vocab = ['<PAD>', '<UNK>', '<EOS>']
  idx = 3
  for w in d:
    vectors[idx, :] = d[w]
    vocab.append(w)
    idx += 1
  lang = sys.argv[4]
  wordvec_name = sys.argv[6]
  outfile_np = 'train_{}.{}.npy'.format(wordvec_name, lang)
  outfile_txt = 'train_vocab.{}.txt'.format(lang)
  np.save(outfile_np, vectors)
  f = open(outfile_txt, 'w')
  print('len(vocab)', len(vocab))
  print('outfile:', outfile_txt, outfile_np)
  for w in vocab:
    f.write(w+'\n')
  f.close()
