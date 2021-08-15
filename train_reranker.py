from env import R2RBatch
from refer360_env import Refer360Batch
from utils import Tokenizer, read_vocab
from vocab import TRAIN_VOCAB
from train import make_arg_parser
from utils import get_arguments
from pprint import pprint
import os
arg_parser = make_arg_parser()
arg_parser.add_argument('--cache_path', type=str,
                        required=True)
args = get_arguments(arg_parser)
vocab = read_vocab(TRAIN_VOCAB, args.language)
tok = Tokenizer(vocab)

if args.env == 'r2r':
  EnvBatch = R2RBatch
elif args.env in ['refer360']:
  EnvBatch = Refer360Batch
if args.prefix in ['refer360', 'r2r', 'R2R', 'REVERIE', 'r360tiny', 'RxR_en-ALL']:
  val_splits = ['val_unseen', 'val_seen']
  target = 'val_unseen'
elif args.prefix in ['touchdown', 'td']:
  val_splits = ['dev']
  target = 'dev'

env = EnvBatch(['none'],
               splits=['train'] + val_splits,
               tokenizer=tok,
               args=args)
if args.env == 'r2r':
  error_margin = 3.0
elif args.env in ['refer360']:
  error_margin = env.distances[0][1] * (2**0.5) + 1


import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
  def __init__(self, input_dim):
    super(Net, self).__init__()
    if args.env == 'r2r':
      self.net = nn.Sequential(
          nn.BatchNorm1d(input_dim),
          nn.Linear(input_dim, input_dim),
          nn.BatchNorm1d(input_dim),
          nn.Tanh(),
          nn.Linear(input_dim, 1)
      )
    elif args.env == 'refer360':
      self.net = nn.Sequential(
          nn.Linear(input_dim, input_dim),
          nn.Tanh(),
          nn.Linear(input_dim, 1)
      )
    else:
      raise NotImplementedError()

  def forward(self, x):

    x = self.net(x).squeeze(-1)
    return x


def average(_l):
  return float(sum(_l)) / len(_l)


def count_prefix_len(l1, l2):
  res = 0
  while(res < len(l1) and res < len(l2) and l1[res] == l2[res]):
    res += 1
  return res


def get_path_len(scanId, path):
  path_len = 0
  prev = path[0]
  for curr in path[1:]:
    if args.env == 'r2r':
      distance = env.distances[scanId][prev][curr]
    elif args.env == 'refer360':
      distance = env.distances[prev][curr]
    else:
      raise NotImplementedError()
    path_len += distance


def load_data(filenames, split_names):
  all_data = {}
  for fn in filenames:

    split = ''
    for split_name in split_names:
      if split_name in fn:
        split = split_name
        break
    assert split != ''

    with open(fn, 'r') as f:
      train_file = json.loads(f.read())
    train_instrs = list(train_file.keys())
    train_data = {}

    for instr_id in train_instrs:
      path_id = instr_id.split('_')[0]
      scanId = env.gt[path_id]['scan']
      new_data = {
          'instr_id': instr_id,
          'candidates': [],
          'candidates_path': [],
          'reranker_inputs': [],
          'distance': [],
          'gt': env.gt[path_id],
          'gold_idx': -1,
          'goal_viewpointId': env.gt[path_id]['path'][-1],
          'gold_len': get_path_len(scanId, env.gt[path_id]['path']),
      }
      self_len = 0
      for i, candidate in enumerate(train_file[instr_id]):
        _, world_states, actions, sum_logits, mean_logits, sum_logp, mean_logp, pm, speaker, scorer = candidate
        new_data['candidates'].append(candidate)
        new_data['candidates_path'].append([ws[1] for ws in world_states])
        new_data['reranker_inputs'].append(
            [len(world_states), sum_logits, mean_logits, sum_logp, mean_logp, pm, speaker] * 4)

        if args.env == 'r2r':
          distance = env.distances[scanId][world_states[-1]
                                           [1]][new_data['goal_viewpointId']]
        elif args.env == 'refer360':
          distance = env.distances[world_states[-1]
                                   [1]][new_data['goal_viewpointId']]
        else:
          raise NotImplementedError()

        new_data['distance'].append(distance)
        my_path = [ws[1] for ws in world_states]
        if my_path == env.gt[path_id]['path']:
          new_data['gold_idx'] = i

      new_data['self_len'] = self_len
      train_data[instr_id] = new_data

    print(fn)
    print('gold', average([d['gold_idx'] != -1 for d in train_data.values()]))
    print('oracle', average(
        [any([dis < error_margin for dis in d['distance']]) for d in train_data.values()]))
    all_data[split] = train_data

  return all_data


cache_list = []
for _f in os.listdir(args.cache_path):
  if 'json' not in _f or 'cache' not in _f:
    continue
  cache_file = os.path.join(args.cache_path, _f)
  cache_list.append(cache_file)

print('Cache list\n')
print('\n'.join(cache_list))
data_splits = load_data(cache_list, ['train'] + val_splits)

net = Net(28).cuda()

batch_labels = []
valid_points = 0

for training_point in data_splits['train'].values():
  labels = training_point['distance']
  gold_idx = np.argmin(labels)
  ac_len = len(labels)
  choice = 1
  x_1 = []
  x_2 = []
  if choice == 1:
    for i in range(ac_len):
      for j in range(ac_len):
        if labels[i] <= error_margin and labels[j] > error_margin:
          x_1.append(i)
          x_2.append(j)
          valid_points += 1
  else:
    for i in range(ac_len):
      if labels[i] > error_margin:
        x_1.append(gold_idx)
        x_2.append(i)
        valid_points += 1
  batch_labels.append((x_1, x_2))

print(valid_points)


x_1 = []
x_2 = []
optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.6)
best_performance = 0.0
for epoch in range(30):  # loop over the dataset multiple times
  epoch_loss = 0
  for i, (instr_id, training_point) in enumerate(data_splits['train'].items()):
    inputs = training_point['reranker_inputs']
    labels = training_point['distance']
    ac_len = len(labels)

    inputs = torch.stack([torch.Tensor(r) for r in inputs]).cuda()
    labels = torch.Tensor(labels)

    scores = net(inputs)

    if i % 10 == 0 and len(x_1):
      x1 = torch.cat(x_1, 0)
      x2 = torch.cat(x_2, 0)
      loss = F.relu(1.0 - (x1 - x2)).mean()
      #s = x1-x2
      #loss = (-s + torch.log(1 + torch.exp(s))).mean()
      loss.backward()
      epoch_loss += loss.item()
      optimizer.step()
      optimizer.zero_grad()
      x_1 = []
      x_2 = []

    if len(batch_labels[i][0]) > 0:
      x_1.append(scores[batch_labels[i][0]])
      x_2.append(scores[batch_labels[i][1]])

  print('epoch', epoch, 'loss', epoch_loss)

  for env_name in ['train'] + val_splits:
    successes = []
    data_dict = data_splits[env_name]
    for instr_id, point in data_dict.items():
      inputs = point['reranker_inputs']
      labels = point['distance']
      inputs = torch.stack([torch.Tensor(r) for r in inputs]).cuda()

      labels = torch.Tensor(labels)
      scores = net(inputs)
      pred = scores.max(0)[1].item()
      successes.append(int(labels[pred] <= error_margin))
    print(env_name, average(successes))
    if env_name is target and average(successes) > best_performance:
      best_performance = average(successes)
      save_path = os.path.join(
          args.cache_path, 'candidates_ranker_{}_{}'.format(env_name, best_performance))
      print('saving to', save_path)
      torch.save(net.state_dict(), save_path)

print('Finished Training')

for env_name in ['train'] + [target]:
  data_dict = data_splits[env_name]
  successes = []
  inspect = [1, 2, 3, 4, 5, 6]
  other_success = [[] for _ in range(len(inspect))]
  spl = []
  for instr_id, point in data_dict.items():
    inputs = point['reranker_inputs']
    labels = point['distance']
    inputs = torch.stack([torch.Tensor(r) for r in inputs]).cuda()
    labels = torch.Tensor(labels)
    scores = net(inputs)

    pred = scores.max(0)[1].item()
    successes.append(int(labels[pred] < error_margin))

    if (int(labels[pred] < error_margin)):
      for i in range(len(point['distance'])):
        pass
        #print( point['reranker_inputs'][i])
        #print( scores[i].item(), point['distance'][i], point['reranker_inputs'][i][5])
      # print("\n")

    for idx, i in enumerate(inspect):
      pred = np.argmax([_input[i] for _input in point['reranker_inputs']])
      other_success[idx].append(int(labels[pred] < error_margin))

  print(env_name, average(successes))
  for idx in range(len(inspect)):
    print(average(other_success[idx]))

perf_name = '{:.4f}'.format(average(successes))
save_path = os.path.join(
    args.cache_path, 'candidates_ranker_{}'.format(perf_name))
print('save_path:', save_path)
torch.save(net.state_dict(), save_path)
