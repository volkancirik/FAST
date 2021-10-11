''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import subprocess
import itertools
import base64
import heapq
from nltk.corpus import wordnet as wn
import torch
import io
import PIL.Image
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('agg')


class DotDict(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']

vocab_pad_idx = base_vocab.index('<PAD>')
vocab_unk_idx = base_vocab.index('<UNK>')
vocab_eos_idx = base_vocab.index('<EOS>')
vocab_bos_idx = base_vocab.index('<BOS>')


def load_nav_graphs(scans):
  ''' Load connectivity graph for each scan '''

  def distance(pose1, pose2):
    ''' Euclidean distance between two graph poses '''
    return ((pose1['pose'][3]-pose2['pose'][3])**2
            + (pose1['pose'][7]-pose2['pose'][7])**2
            + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

  graphs = {}
  file_path = os.path.dirname(__file__)
  connect_folder = os.path.abspath(
      os.path.join(file_path, '..', '..', 'connectivity'))
  for scan in scans:
    with open('%s/%s_connectivity.json' % (connect_folder, scan)) as f:
      G = nx.Graph()
      positions = {}
      data = json.load(f)
      for i, item in enumerate(data):
        if item['included']:
          for j, conn in enumerate(item['unobstructed']):
            if conn and data[j]['included']:
              positions[item['image_id']] = np.array([item['pose'][3],
                                                      item['pose'][7], item['pose'][11]])
              assert data[j]['unobstructed'][i], 'Graph should be undirected'
              G.add_edge(item['image_id'], data[j]['image_id'],
                         weight=distance(item, data[j]))
      nx.set_node_attributes(G, values=positions, name='position')
      graphs[scan] = G
  return graphs


def load_datasets(splits,
                  prefix='R2R'
                  ):
  data = []
  file_path = os.path.dirname(__file__)
  for split in splits:
    _path = os.path.abspath(os.path.join(
        file_path, 'data', '{}_{}.json'.format(prefix, split)))
    print('loading from:', _path)
    with open(_path) as f:
      data += json.load(f)
  return data


def decode_base64(string):
  if sys.version_info[0] == 2:
    return base64.decodestring(bytearray(string))
  elif sys.version_info[0] == 3:
    return base64.decodebytes(bytearray(string, 'utf-8'))
  else:
    raise ValueError(
        'decode_base64 can not handle python version {}'.format(sys.version_info[0]))


class Tokenizer(object):
  ''' Class to tokenize and encode a sentence. '''
  SENTENCE_SPLIT_REGEX = re.compile(
      r'(\W+)')  # Split on any non-alphanumeric character

  def __init__(self, vocab=None):
    self.vocab = vocab
    self.word_to_index = {}
    self.index_is_verb = {}
    if vocab:
      for i, word in enumerate(vocab):
        self.word_to_index[word] = i
        self.index_is_verb[i] = int(Tokenizer.is_verb(word))

  @staticmethod
  def split_sentence(sentence, language='en-OLD'):
    ''' Break sentence into a list of words and punctuation '''
    toks = []
    if 'en' in language:
      tokens = [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(
          sentence.strip()) if len(s.strip()) > 0]
    else:
      tokens = [s.strip().lower() for s in sentence.split()]
    for word in tokens:
      # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
      if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
        toks += list(word)
      else:
        toks.append(word)
    return toks

  def filter_verb(self, toks, sel_verb=False):
    is_verb = [self.index_is_verb[tok] for tok in toks]
    if sel_verb:
      sel_indexes = [i for i, x in enumerate(is_verb) if x]
    else:
      sel_indexes = [i for i, x in enumerate(is_verb) if not x]
    return is_verb, sel_indexes

  @staticmethod
  def is_verb(word):
    if word in base_vocab:
      return True
    for _entry in wn.synsets(word):
      if _entry.name().split('.')[0] == word and _entry.pos() == 'v':
        return True
    return False

  def encode_sentence(self, sentence, language='en-OLD'):
    if len(self.word_to_index) == 0:
      sys.exit('Tokenizer has no vocab')
    encoding = []

    n_unk, n_found, unk = 0., 0., set()
    for word in Tokenizer.split_sentence(sentence,
                                         language=language):
      if word in self.word_to_index:
        encoding.append(self.word_to_index[word])
        n_found += 1.0
      else:
        encoding.append(vocab_unk_idx)
        n_unk += 1.0
        unk.add(word)
    # encoding.append(vocab_eos_idx)
    #utterance_length = len(encoding)
    # if utterance_length < self.encoding_length:
      #encoding += [vocab_pad_idx] * (self.encoding_length - len(encoding))
    # encoding = encoding[:self.encoding_length] # leave room for unks

    arr = np.array(encoding)
    return arr, len(encoding), n_unk, n_found, unk

  def decode_sentence(self, encoding, break_on_eos=False, join=True):
    sentence = []
    for ix in encoding:
      if ix == (vocab_eos_idx if break_on_eos else vocab_pad_idx):
        break
      else:
        sentence.append(self.vocab[ix])
    if join:
      return ' '.join(sentence)
    return sentence


def build_vocab(splits=['train'],
                min_count=5,
                start_vocab=base_vocab,
                prefix='R2R'):
  ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
  count = Counter()
  data = load_datasets(splits, prefix=prefix)
  for item in data:
    for instr in item['instructions']:
      count.update(Tokenizer.split_sentence(instr))
  vocab = list(start_vocab)
  for word, num in count.most_common():
    if num >= min_count:
      vocab.append(word)
    else:
      break
  return vocab


def write_vocab(vocab, path):
  print('Writing vocab of size %d to %s' % (len(vocab), path))
  with open(path, 'w') as f:
    for word in vocab:
      f.write('%s\n' % word)


def read_vocab(path, language):
  vocab = []
  with open(path + '{}.txt'.format(language)) as f:
    vocab = [word.strip() for word in f.readlines()]
  return vocab


def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)


def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def k_best_indices(arr, k, sorted=False):
  # https://stackoverflow.com/a/23734295
  if k >= len(arr):
    if sorted:
      return np.argsort(arr)
    else:
      return np.arange(0, len(arr))
  ind = np.argpartition(arr, -k)[-k:]
  if sorted:
    ind = ind[np.argsort(arr[ind])]
  return ind


def structured_map(function, *args, **kwargs):
  #assert all(len(a) == len(args[0]) for a in args[1:])
  nested = kwargs.get('nested', False)
  acc = []
  for t in zip(*args):
    if nested:
      mapped = [function(*inner_t) for inner_t in zip(*t)]
    else:
      mapped = function(*t)
    acc.append(mapped)
  return acc


def flatten(lol):
  return [l for lst in lol for l in lst]


def all_equal(lst):
  return all(x == lst[0] for x in lst[1:])


def try_cuda(pytorch_obj):
  import torch.cuda
  try:
    disabled = torch.cuda.disabled
  except:
    disabled = False
  if torch.cuda.is_available() and not disabled:
    return pytorch_obj.cuda()
  else:
    return pytorch_obj


def pretty_json_dump(obj, fp):
  json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))


def spatial_feature_from_bbox(bboxes, im_h, im_w):
  # from Ronghang Hu
  # https://github.com/ronghanghu/cmn/blob/ff7d519b808f4b7619b17f92eceb17e53c11d338/models/spatial_feat.py

  # Generate 5-dimensional spatial features from the image
  # [xmin, ymin, xmax, ymax, S] where S is the area of the box
  if isinstance(bboxes, list):
    bboxes = np.array(bboxes)
  bboxes = bboxes.reshape((-1, 4))
  # Check the size of the bounding boxes
  assert np.all(bboxes[:, 0:2] >= 0)
  assert np.all(bboxes[:, 0] <= bboxes[:, 2])
  assert np.all(bboxes[:, 1] <= bboxes[:, 3])
  assert np.all(bboxes[:, 2] <= im_w)
  assert np.all(bboxes[:, 3] <= im_h)

  feats = np.zeros((bboxes.shape[0], 5), dtype=np.float32)
  feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
  feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
  feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
  feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
  feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1])  # S
  return feats


def get_arguments(arg_parser):
  arg_parser.add_argument('--pdb', action='store_true')
  arg_parser.add_argument('--ipdb', action='store_true')
  arg_parser.add_argument('--no_cuda', action='store_true')
  arg_parser.add_argument('--experiment_name', type=str, default='debug')
  arg_parser.add_argument('--batch_size', type=int, default=64)
  arg_parser.add_argument('--save_args', action='store_false')

  args = arg_parser.parse_args()
  args.image_list_file = os.path.join(args.refer360_root, 'imagelist.txt')
  args.butd_filename = os.path.join(
      args.img_features_root, '{}_{}degrees_obj36.tsv'.format(args.prefix, args.angle_inc))
  args.refer360_data = os.path.join(args.refer360_root,
                                    'continuous_grounding_{}degrees'.format(args.angle_inc))
  args.cache_root = os.path.join(args.refer360_root,
                                 'cached_data_{}degrees'.format(args.angle_inc))
  args.prior_prefix = os.path.join(
      args.img_features_root, '{}_{}degrees_'.format(args.prefix, args.angle_inc))

  if args.prefix in ['refer360', 'r360tiny', 'touchdown', 'td']:
    args.env = 'refer360'
    args.language = 'refer360big' if not args.language else args.language

    if args.metrics == 'success':
      args.metrics = 'fov_accuracy'
    if args.blind:
      args.refer360_image_feature_type = ['none']
  else:
    args.env = 'r2r'
    args.language = 'en-ALL'

  if args.use_reading:
    args.feedback_method = 'teacher'
    args.use_gt_actions = True
  args.wordvec_path += '.{}.npy'.format(args.language)
  return args


def run(arg_parser, entry_function, functions=None):

  args = get_arguments(arg_parser)
  print('parameters:')
  print(json.dumps(vars(args), indent=2))

  args = DotDict(vars(args))

  args.RESULT_DIR = os.path.join(
      'tasks/{}/experiments/'.format(args.prefix), args.experiment_name, 'results')
  args.SNAPSHOT_DIR = os.path.join(
      'tasks/{}/experiments/'.format(args.prefix), args.experiment_name, 'snapshots')
  args.PLOT_DIR = os.path.join(
      'tasks/{}/experiments/'.format(args.prefix), args.experiment_name, 'plots')

  make_dirs([args.RESULT_DIR, args.SNAPSHOT_DIR, args.PLOT_DIR])

  import torch.cuda
  torch.cuda.disabled = args.no_cuda

  if entry_function == None:
    print('will run', args.function)
    entry_function = functions[args.function]
  if args.ipdb:
    import ipdb
    ipdb.runcall(entry_function, args)
  elif args.pdb:
    import pdb
    pdb.runcall(entry_function, args)
  else:
    entry_function(args)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color='green', bold=False, highlight=False):
  attr = []
  num = color2num[color]
  if highlight:
    num += 10
  attr.append(str(num))
  if bold:
    attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def filter_param(m):
  return [p for p in m.parameters() if p.requires_grad]


def module_grad(module, requires_grad=False):
  for p in module.parameters():
    p.requires_grad_(requires_grad)


def make_dirs(list_of_dirs):
  for directory in list_of_dirs:
    if not os.path.exists(directory):
      os.makedirs(directory)


class PriorityQueue:
  def __init__(self, max_size=0, maxHeap=True):
    self.queue = []
    self.priority = []
    self.pri = []
    self.maxHeap = maxHeap
    self.locked = False
    self.len = 0
    assert(max_size == 0)

  def lock(self):
    self.locked = True

  def push(self, item, priority):
    if self.locked:
      return
    self.queue.append(item)
    self.priority.append(priority)

    p = priority.item() if type(priority) is torch.Tensor else priority
    if self.maxHeap:
      p = - p
    heapq.heappush(self.pri, (p, self.len))
    self.len += 1

  def pop(self):
    if self.locked:
      return 0, self.priority[0], self.queue[0]
    if len(self.pri) == 0:
      print('PriorityQueue error: pop from an empty queue')
      import pdb
      pdb.set_trace()
    p, idx = heapq.heappop(self.pri)
    item = self.queue[idx]
    priority = self.priority[idx]
    return idx, priority, item

  def peak(self):
    if len(self.pri) == 0:
      return None
    p, idx = self.pri[0]
    return idx, self.priority[idx], self.queue[idx]

  def size(self):
    return len(self.pri)


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, torch.Tensor):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def get_confusion_matrix_image(labels, matrix, title='Title', tight=False, cmap=cm.copper):

  labels_x, labels_y = labels
  fig, ax = plt.subplots()
  _ = ax.imshow(matrix, cmap=cmap)

  ax.set_xticks(np.arange(matrix.shape[1]))
  ax.set_yticks(np.arange(matrix.shape[0]))
  ax.set_xticklabels(labels_x)
  ax.set_yticklabels(labels_y)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor')

  # Loop over data dimensions and create text annotations.
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      _ = ax.text(j, i, '{:0.2f}'.format(matrix[i, j]),
                  ha='center', va='center', color='w', fontsize=8)
  ax.set_title(title)
  if tight:
    fig.tight_layout()
  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = ToTensor()(image)
  plt.close('all')
  return image


def get_bar_image(x_pos, x_labels, means, errors,
                  tight=False,
                  title='Title'):
  plt.figure(figsize=(24, 12))
  fig, ax = plt.subplots()

  ax.bar(x_pos, means, yerr=errors, align='center',
         alpha=0.5, ecolor='black', capsize=10)
  ax.set_xticks(x_pos)
  ax.set_xticklabels(x_labels)
  ax.yaxis.grid(True)
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
           rotation_mode='anchor', fontsize=9)

  ax.set_title(title)
  if tight:
    fig.tight_layout()

  buf = io.BytesIO()
  plt.savefig(buf, format='jpeg')
  buf.seek(0)
  image = PIL.Image.open(buf)
  image = ToTensor()(image)
  plt.close('all')
  return image
