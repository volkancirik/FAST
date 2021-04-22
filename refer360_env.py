''' Batched Refer360 grounding environment '''
import sys
import os
import copy
import itertools
import csv
import pdb
import random
import os.path

from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from env import _build_action_embedding
from env import R2RBatch
from utils import structured_map, try_cuda
from refer360_sim import Refer360Simulator, WorldState
import base64
file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(file_path, '..', '..', 'build'))
sys.path.append(module_path)
csv.field_size_limit(sys.maxsize)

DIR2IDX = {
    'ul': 0,
    'u': 1,
    'ur': 2,
    'l': 3,
    'r': 5,
    'dl': 6,
    'd': 7,
    'dr': 8
}


def build_viewpoint_loc_embedding(viewIndex,
                                  angle_inc=15.0):
  """
  Position embedding:
  heading 64D + elevation 64D
  1) heading: [sin(heading) for _ in range(1, 9)] +
              [cos(heading) for _ in range(1, 9)]
  2) elevation: [sin(elevation) for _ in range(1, 9)] +
                [cos(elevation) for _ in range(1, 9)]
  """
  embedding = np.zeros((9, 128), np.float32)

  for absViewIndex in range(9):
    relViewIndex = (
        absViewIndex) % 3 + (absViewIndex // 3) * 3
    rel_heading = (relViewIndex % 3 - viewIndex % 3) * angle_inc
    rel_elevation = ((relViewIndex // 3) - viewIndex // 3) * angle_inc
    embedding[absViewIndex,  0:32] = np.sin(rel_heading)
    embedding[absViewIndex, 32:64] = np.cos(rel_heading)
    embedding[absViewIndex, 64:96] = np.sin(rel_elevation)
    embedding[absViewIndex,   96:] = np.cos(rel_elevation)
  return embedding


def _build_visited_embedding(adj_loc_list, visited):
  n_emb = 64
  half = int(n_emb/2)
  embedding = np.zeros((len(adj_loc_list), n_emb), np.float32)
  for kk, adj in enumerate(adj_loc_list):
    val = visited[adj['nextViewpointId']]
    embedding[kk,  0:half] = np.sin(val)
    embedding[kk, half:] = np.cos(val)
  return embedding


def _build_oracle_embedding(adj_loc_list, gt_viewpoint_idx):
  n_emb = 64
  half = int(n_emb/2)
  embedding = np.zeros((len(adj_loc_list), n_emb), np.float32)

  for kk, adj in enumerate(adj_loc_list):
    val = 0
    if kk == gt_viewpoint_idx:
      val = 1
    embedding[kk,  0:half] = np.sin(val)
    embedding[kk, half:] = np.cos(val)
  return embedding


def load_world_state(sim, world_state):
  sim.newEpisode(world_state)


def get_world_state(sim):
  state = sim.getState()
  return WorldState(scanId=state.scanId,
                    viewpointId=state.viewpointId,
                    heading=state.heading,
                    elevation=state.elevation,
                    viewIndex=state.viewIndex)


def _navigate_to_location(sim, nextViewpointId):
  state = sim.getState()
  if state.viewpointId == nextViewpointId:
    return
  sim.look_fov(nextViewpointId)


def _get_panorama_states(sim):
  '''
  Look around and collect all the navigable locations
  '''
  state = sim.getState()
  adj_list = sim.get_neighbors()

  adj_dict = {}
  for adj in adj_list:
    adj_dict[adj['idx']] = {
        'absViewIndex': DIR2IDX[sim.nodes[state.viewpointId]['neighbor2dir'][adj['idx']]],
        'nextViewpointId': adj['idx'],
        'rel_heading': adj['lng'] - state.heading,
        'rel_elevation': adj['lat'] - state.elevation,
        'distance': sim.distances[adj['idx']][state.viewpointId],
    }

  stop = {
      'absViewIndex': -1,
      'nextViewpointId': state.viewpointId
  }
  adj_loc_list = [stop] + sorted(
      adj_dict.values(), key=lambda x: abs(x['rel_heading']))

  return state, adj_loc_list


def make_sim(cache_root='',
             image_w=400,
             image_h=400,
             fov=90):
  sim = Refer360Simulator(cache_root,
                          output_image_shape=(image_h, image_w),
                          fov=fov)
  return sim


class Refer360ImageFeatures(object):
  NUM_VIEWS = 9
  MEAN_POOLED_DIM = 2048
  feature_dim = MEAN_POOLED_DIM

  IMAGE_W = 400
  IMAGE_H = 400
  VFOV = 90

  @staticmethod
  def from_args(args):
    feats = []

    n_fovs = int((360 / args.angle_inc)*(150/args.angle_inc))
    for image_feature_type in sorted(args.refer360_image_feature_type):
      if 'none' in image_feature_type:
        feats.append(NoImageFeatures())
      if 'random' in image_feature_type:
        feats.append(RandImageFeatures())
      if 'mean_pooled' in image_feature_type:
        feats.append(MeanPooledImageFeatures(cache_root=args.cache_root,
                                             image_list_file=args.image_list_file,
                                             n_fovs=n_fovs,
                                             feature_model=args.refer360_image_feature_model))
      if 'butd' in image_feature_type:
        feats.append(BUTDImageFeatures(cache_root=args.cache_root,
                                       image_list_file=args.image_list_file,
                                       butd_filename=args.butd_filename,
                                       n_fovs=n_fovs,))

      assert len(feats) >= 1
    return feats

  @staticmethod
  def add_args(argument_parser):
    argument_parser.add_argument("--refer360_image_feature_type", nargs="+",
                                 choices=['none', 'random',
                                          'mean_pooled', 'butd'],
                                 default=['mean_pooled'])
    argument_parser.add_argument("--refer360_image_feature_model",
                                 choices=['resnet', 'clip'],
                                 default='resnet')

  def get_name(self):
    raise NotImplementedError("base class does not have get_name")

  def batch_features(self, feature_list):
    features = np.stack(feature_list)
    return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

  def get_features(self, state):
    raise NotImplementedError("base class does not have get_features")


class RandImageFeatures(Refer360ImageFeatures):
  feature_dim = 24

  def __init__(self):
    print('Random image features will be provided')
    self.features = np.random.randn(
        Refer360ImageFeatures.NUM_VIEWS, self.feature_dim)

  def get_features(self, state):
    return self.features

  def get_name(self):
    return "random"


class NoImageFeatures(Refer360ImageFeatures):
  feature_dim = Refer360ImageFeatures.MEAN_POOLED_DIM

  def __init__(self):
    print('Image features not provided')
    self.features = np.zeros(
        (Refer360ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)

  def get_features(self, state):
    return self.features

  def get_name(self):
    return "none"


MODEL2PREFIX = {'resnet': '',
                'clip': '.clip'}
MODEL2FEATURE_DIM = {'resnet': 2048,
                     'clip': 512}


class MeanPooledImageFeatures(Refer360ImageFeatures):
  def __init__(self,
               cache_root='',
               image_list_file='',
               feature_model='resnet',
               n_fovs=240):
    self.feature_dim = MODEL2FEATURE_DIM[feature_model]
    self.feature_model = MODEL2PREFIX[feature_model]
    self.features = {}
    self.n_fovs = n_fovs

    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    nodes = meta['nodes']

    print('loading image features for refer360 from', image_list_file)
    image_list = [line.strip()
                  for line in open(image_list_file)]

    pbar = tqdm(image_list)

    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]
      feature_file = os.path.join(
          cache_root, 'features', '{}'.format(pano) + self.feature_model + '.npy')
      if not os.path.exists(feature_file):
        print('file missing:', feature_file)
        quit(0)

      fov_feats = np.load(feature_file).squeeze()

      for idx in range(self.n_fovs):
        feats = np.zeros(
            (Refer360ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)
        feats[4, :] = fov_feats[idx]

        for neighbor in nodes[idx]['neighbor2dir']:
          n_direction = nodes[idx]['neighbor2dir'][neighbor]
          dir_idx = DIR2IDX[n_direction]
          feats[dir_idx, :] = fov_feats[neighbor]
        pano_fov = self._make_id(pano, idx)
        self.features[pano_fov] = feats

  def _make_id(self, scanId, viewpointId):
    return '{}'.format(scanId) + '_' + '{}'.format(viewpointId)

  def get_features(self, state):
    long_id = self._make_id(state.scanId, state.viewpointId)
    # Return feature of all the 36 views
    return self.features[long_id]

  def get_name(self):
    name = "mean_pooled"+self.feature_model
    return name


class BUTDImageFeatures(Refer360ImageFeatures):
  def __init__(self, butd_filename='',
               cache_root='',
               image_list_file='',
               n_fovs=240):

    print('Loading bottom-up top-down features')
    self.features = defaultdict(list)
    self.feature_dim = 2048
    self.n_fovs = n_fovs

    FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                  "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
    self.features = {}

    fov2feat = {}
    with open(butd_filename) as f:
      reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
      for i, item in enumerate(reader):

        for key in ['img_h', 'img_w', 'num_boxes']:
          item[key] = int(item[key])

        boxes = item['num_boxes']
        decode_config = [
            ('objects_id', (boxes, ), np.int64),
            ('objects_conf', (boxes, ), np.float32),
            ('attrs_id', (boxes, ), np.int64),
            ('attrs_conf', (boxes, ), np.float32),
            ('boxes', (boxes, 4), np.float32),
            ('features', (boxes, -1), np.float32),
        ]
        for key, shape, dtype in decode_config:
          item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
          item[key] = item[key].reshape(shape)
          item[key].setflags(write=False)

        scanId, viewpointId = item['img_id'].split('.')

        pano_fov = self._make_id(scanId, viewpointId)
        feats = np.sum(item['features'], axis=0)
        fov2feat[pano_fov] = feats

    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    nodes = meta['nodes']

    print('loaded BUTD features', image_list_file)
    print('preparing image features for refer360..')
    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]

      for idx in range(self.n_fovs):
        feats = np.zeros(
            (Refer360ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)
        feats[4, :] = fov2feat['{}_{}'.format(pano, idx)]

        for neighbor in nodes[idx]['neighbor2dir']:
          n_direction = nodes[idx]['neighbor2dir'][neighbor]
          dir_idx = DIR2IDX[n_direction]
          feats[dir_idx, :] = fov2feat['{}_{}'.format(pano, neighbor)]
        pano_fov = self._make_id(pano, idx)
        self.features[pano_fov] = feats
    print('image features for refer360 are prepared.')

  def _make_id(self, scanId, viewpointId):
    return '{}_{}'.format(scanId, viewpointId)

  def get_features(self, state):
    long_id = self._make_id(state.scanId, state.viewpointId)
    return self.features[long_id]

  def get_name(self):
    name = "_with_butd"
    return name


class Refer360EnvBatch():
  ''' A simple wrapper for a batch of MatterSim environments,
      using discretized viewpoints and pretrained features '''

  def __init__(self, batch_size, beam_size,
               cache_root=''):
    self.sims = []
    self.batch_size = batch_size
    self.beam_size = beam_size
    self.cache_root = cache_root
    sim = make_sim(cache_root,
                   Refer360ImageFeatures.IMAGE_W,
                   Refer360ImageFeatures.IMAGE_H,
                   Refer360ImageFeatures.VFOV)

    for i in range(batch_size):
      beam = []
      for j in range(beam_size):
        beam.append(copy.deepcopy(sim))
      self.sims.append(beam)

  def sims_view(self, beamed):
    if beamed:
      return [itertools.cycle(sim_list) for sim_list in self.sims]
    else:
      return (s[0] for s in self.sims)

  def newEpisodes(self, scanIds, viewpointIds, headings, beamed=False):
    assert len(scanIds) == len(viewpointIds)
    assert len(headings) == len(viewpointIds)
    assert len(scanIds) == len(self.sims)
    world_states = []
    for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
      world_state = WorldState(scanId, viewpointId, heading, 0, 0)
      if beamed:
        world_states.append([world_state])
      else:
        world_states.append(world_state)
      load_world_state(self.sims[i][0], world_state)
    assert len(world_states) == len(scanIds)
    return world_states

  def getStates(self, world_states, beamed=False):
    ''' Get list of states. '''
    def f(sim, world_state):
      load_world_state(sim, world_state)
      return _get_panorama_states(sim)
    return structured_map(f, self.sims_view(beamed), world_states, nested=beamed)

  def makeActions(self, world_states, actions, last_obs, beamed=False):
    ''' Take an action using the full state dependent action interface (with batched input).
        Each action is an index in the adj_loc_list,
        0 means staying still (i.e. stop)
    '''
    def f(sim, world_state, action, last_ob):
      load_world_state(sim, world_state)
      # load the location attribute corresponding to the action
      if action >= len(last_ob['adj_loc_list']):
        pdb.set_trace()
      loc_attr = last_ob['adj_loc_list'][action]
      _navigate_to_location(
          sim, loc_attr['nextViewpointId'])
      # sim.makeAction(index, heading, elevation)
      return get_world_state(sim)
    return structured_map(f, self.sims_view(beamed), world_states, actions, last_obs, nested=beamed)


r2r2Refer360 = {'val_seen': 'validation.seen',
                'val_unseen': 'validation.unseen',
                'test_seen': 'test.seen',
                'test_unseen': 'test.unseen',
                'train': 'train'
                }


def load_datasets(splits,
                  root='',
                  use_intermediate=True):
  d, s = [], []
  converted = []
  act_length = []
  sen_length = []
  for split_name in splits:
    fname = os.path.join(
        root, '{}.[{}].imdb.npy'.format(r2r2Refer360[split_name], 'all'))
    print('loading split from {}'.format(fname))
    dump = np.load(fname, allow_pickle=True)[()]
    d += dump['data_list'][0]

  if use_intermediate and 'train' in splits:
    print('will use intermediate paths for training')
  for datum in d:
    if len(datum['path']) <= 1:
      continue

    datum['path_id'] = datum['annotationid']
    datum['scan'] = datum['img_src'].split('/')[-1].split('.')[0]
    datum['heading'] = 0
    datum['elevation'] = 0

    if use_intermediate and split_name == 'train':
      for kk, refexp in enumerate(datum['refexps']):
        new_datum = datum.copy()
        instructions = " ".join(refexp)
        new_datum['instructions'] = [instructions]
        new_datum['path'] = datum['intermediate_paths'][kk]
        new_datum['gt_actions_path'] = datum['intermediate_paths'][kk]
        if len(new_datum['path']) <= 1:
          continue
        act_length += [len(new_datum['path'])]
        sen_length += [len(new_datum['instructions'][0].split(' '))]

        converted.append(new_datum)
    else:
      instructions = " ".join([" ".join(refexp)
                               for refexp in datum['refexps']]).replace('.', ' . ').replace(',', ' , ').replace(';', ' ; ')
      datum['instructions'] = [instructions]
      datum['gt_actions_path'] = datum['path']
      act_length += [len(datum['path'])]
      sen_length += [len(datum['instructions'][0].split(' '))]

      converted.append(datum)
  print('min max mean path length: {:2.2f} {:2.2f} {:2.2f}'.format(np.min(act_length),
                                                                   np.max(
      act_length),
      np.mean(act_length)))
  print('min max mean instruction length: {:2.2f} {:2.2f} {:2.2f}'.format(np.min(sen_length),
                                                                          np.max(
      sen_length),
      np.mean(sen_length)))
  print('# of instances:', len(converted))
  return converted


class Refer360Batch(R2RBatch):
  ''' Implements the Refer360 grounding task, using discretized viewpoints and pretrained features '''

  def __init__(self, image_features_list,
               splits=['train'],
               tokenizer=None,
               instruction_limit=None,
               args=None):
    batch_size = args.batch_size
    seed = args.seed
    beam_size = args.beam_size
    language = args.language
    refer360_root = args.refer360_root
    cache_root = args.cache_root
    use_intermediate = args.use_intermediate
    use_gt_actions = args.use_gt_actions
    use_visited_embeddings = args.use_visited_embeddings
    use_oracle_embeddings = args.use_oracle_embeddings
    angle_inc = args.angle_inc

    self.num_views = Refer360ImageFeatures.NUM_VIEWS
    self.angle_inc = angle_inc
    self.image_features_list = image_features_list
    self.data = []
    self.scans = []
    self.gt = {}
    self.tokenizer = tokenizer

    counts = defaultdict(int)

    print('loading splits:', splits)
    refer360_data = load_datasets(splits,
                                  root=refer360_root,
                                  use_intermediate=use_intermediate)

    total_unk, total_found, all_unk = 0, 0, set()
    for item in refer360_data:
      path_id = item['path_id']
      count = counts[path_id]
      new_path_id = '{}*{}'.format(path_id, count)
      counts[path_id] += 1
      item['path_id'] = new_path_id

      self.gt[item['path_id']] = item
      instructions = item['instructions']
      if instruction_limit:
        instructions = instructions[:instruction_limit]

      for j, instr in enumerate(instructions):
        self.scans.append(item['scan'])
        new_item = dict(item)
        new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
        new_item['instructions'] = instr
        if tokenizer:
          self.tokenizer = tokenizer
          new_item['instr_encoding'], new_item['instr_length'], n_unk, n_found, unk = tokenizer.encode_sentence(
              instr, language=language)
          total_found += n_found
          total_unk += n_unk
          all_unk |= unk
        else:
          self.tokenizer = None
        self.data.append(new_item)
    print('unk ratio: {:3.2f} {} {}'.format(
        total_unk / (total_unk + total_found + 1), total_unk, total_found))
    print('UNK vocab size:', len(all_unk))
    if args.verbose:
      print('UNK vocab:\n', all_unk)
    self.scans = set(self.scans)
    self.splits = splits
    self.seed = seed
    random.seed(self.seed)
    random.shuffle(self.data)
    self.instr_id_to_idx = {}
    for i, item in enumerate(self.data):
      self.instr_id_to_idx[item['instr_id']] = i
    self.ix = 0
    self.batch_size = batch_size
    self.set_beam_size(beam_size,
                       cache_root=cache_root)
    self.print_progress = False
    print('Refer360Batch loaded with %d instructions, using splits: %s' %
          (len(self.data), ",".join(splits)))
    self.batch = self.data
    self.notTest = ('test' not in splits)
    self.paths = self.env.sims[0][0].paths
    self.distances = self.env.sims[0][0].distances
    if use_gt_actions:
      self._action_fn = self._gt_action
      print('will use ground-truth actions')
    else:
      self._action_fn = self._shortest_path_action
      print('will use shortest path actions')
    self.use_visited_embeddings = use_visited_embeddings
    self.use_oracle_embeddings = use_oracle_embeddings
    self._static_loc_embeddings = [
        build_viewpoint_loc_embedding(viewIndex, angle_inc=self.angle_inc) for viewIndex in range(9)]

  def set_beam_size(self, beam_size,
                    force_reload=False,
                    cache_root=''):
    # warning: this will invalidate the environment, self.reset() should be called afterward!
    try:
      invalid = (beam_size != self.beam_size)
    except:
      invalid = True
    if force_reload or invalid:
      self.beam_size = beam_size
      self.env = Refer360EnvBatch(self.batch_size, beam_size,
                                  cache_root=cache_root)

  def _gt_action(self, state, adj_loc_list, goalViewpointId, gt_path):
    '''
    Determine next action on the grount-truth path to goal,
    for supervised training.
    '''
    if len(gt_path) == 1:
      return 0, gt_path
    assert state.viewpointId == gt_path[0], "state.viewpointId != gt_path[0] {} != {} {} {} {}".format(
        state.viewpointId, gt_path[0], gt_path, adj_loc_list, state)
    nextViewpointId = gt_path[1]
    for n_a, loc_attr in enumerate(adj_loc_list):
      if loc_attr['nextViewpointId'] == nextViewpointId:
        return n_a, gt_path[1:]
    # Next nextViewpointId not found! This should not happen!
    print('adj_loc_list:', adj_loc_list)
    print('nextViewpointId:', nextViewpointId)
    long_id = '{}_{}'.format(state.scanId, state.viewpointId)
    print('longId:', long_id)
    raise Exception('Bug: nextViewpointId not in adj_loc_list')

  def _shortest_path_action(self, state, adj_loc_list, goalViewpointId, gt_path):
    '''
    Determine next action on the shortest path to goal,
    for supervised training.
    '''
    if state.viewpointId == goalViewpointId:
      return 0, gt_path  # do nothing
    path = self.paths[state.viewpointId][
        goalViewpointId]
    nextViewpointId = path[1]
    for n_a, loc_attr in enumerate(adj_loc_list):
      if loc_attr['nextViewpointId'] == nextViewpointId:
        return n_a, gt_path
    # Next nextViewpointId not found! This should not happen!
    print('adj_loc_list:', adj_loc_list)
    print('nextViewpointId:', nextViewpointId)
    long_id = '{}_{}'.format(state.scanId, state.viewpointId)
    print('longId:', long_id)
    raise Exception('Bug: nextViewpointId not in adj_loc_list')

  def _deviation(self, state, given_path):
    all_paths = self.paths[state.viewpointId]
    near_id = given_path[0]
    near_d = len(all_paths[near_id])
    for item in given_path:
      d = len(all_paths[item])
      if d < near_d:
        near_id = item
        near_d = d
    return near_d - 1  # MUST - 1

  def _distance(self, state, given_path):
    goalViewpointId = given_path[-1]
    return self.distances[state.viewpointId][goalViewpointId]

  def _progress(self, state, given_path):
    goalViewpointId = given_path[-1]
    if state.viewpointId == goalViewpointId:
      return 1.0
    given_path_len = len(given_path) - 1
    path = self.paths[state.viewpointId][
        goalViewpointId]
    path_len = len(path) - 1
    return 1.0 - float(path_len) / given_path_len

  def observe(self, world_states, beamed=False, include_teacher=True, instr_id=None):
    # start_time = time.time()
    obs = []
    for i, states_beam in enumerate(self.env.getStates(world_states, beamed=beamed)):
      item = self.batch[i]
      obs_batch = []
      for state, adj_loc_list in states_beam if beamed else [states_beam]:
        if item['scan'] != state.scanId:
          item = self.data[self.instr_id_to_idx[instr_id]]
          assert item['scan'] == state.scanId
        feature = [featurizer.get_features(state)
                   for featurizer in self.image_features_list]
        # assert len(feature) == 1, 'for now, only work with MeanPooled feature'
        if len(feature) == 1:
          feature_with_loc = np.concatenate(
              (feature[0], self._static_loc_embeddings[state.viewIndex]), axis=-1)
        elif len(feature) == 2:
          feature_with_loc = np.concatenate(
              (feature[0], feature[1], self._static_loc_embeddings[state.viewIndex]), axis=-1)
        else:
          raise NotImplementedError(
              'for now, only work with MeanPooled feature or with Rand features')
        action_embedding = _build_action_embedding(adj_loc_list, feature)

        teacher_action, new_path = self._action_fn(
            state, adj_loc_list, item['gt_actions_path'][-1], item['gt_actions_path'])
        if self.use_visited_embeddings:
          item['visited_viewpoints'][state.viewpointId] += 1.0
          visited_embedding = _build_visited_embedding(
              adj_loc_list, item['visited_viewpoints'])
          action_embedding = np.concatenate(
              (action_embedding, visited_embedding), axis=-1)
        if self.use_oracle_embeddings:
          oracle_embedding = _build_oracle_embedding(
              adj_loc_list, teacher_action)
          action_embedding = np.concatenate(
              (action_embedding, oracle_embedding), axis=-1)

        ob = {
            'instr_id': item['instr_id'],
            'scan': state.scanId,
            'viewpoint': state.viewpointId,
            'viewIndex': state.viewIndex,
            'heading': state.heading,
            'elevation': state.elevation,
            'feature': [feature_with_loc],
            'adj_loc_list': adj_loc_list,
            'action_embedding': action_embedding,
            'instructions': item['instructions'],
        }
        if include_teacher and self.notTest:
          ob['teacher'] = teacher_action
          self.batch[i]['gt_actions_path'] = new_path
          ob['deviation'] = self._deviation(state, item['path'])
          ob['progress'] = self._progress(state, item['path']),
          ob['distance'] = self._distance(state, item['path']),
        if 'instr_encoding' in item:
          ob['instr_encoding'] = item['instr_encoding']
        if 'instr_length' in item:
          ob['instr_length'] = item['instr_length']
        obs_batch.append(ob)
      if beamed:
        obs.append(obs_batch)
      else:
        assert len(obs_batch) == 1
        obs.append(obs_batch[0])
    return obs
