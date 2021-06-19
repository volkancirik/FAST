''' Batched Refer360 grounding environment '''
import sys
import os
import copy
import itertools
import csv
import pdb
import random
import os.path
import base64
import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from env import _build_action_embedding
from env import R2RBatch
from utils import structured_map, try_cuda
from refer360_sim import Refer360Simulator, WorldState
from refer360_utils import DIR2IDX, MODEL2FEATURE_DIM, MODEL2PREFIX
from refer360_utils import get_object_dictionaries
from refer360_utils import load_cnn_features, load_vectors, load_butd
from refer360_utils import build_viewpoint_loc_embedding
from refer360_utils import build_visited_embedding
from refer360_utils import build_oracle_embedding

file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(
    file_path, '..', '..', 'build_refer360'))
sys.path.append(module_path)
csv.field_size_limit(sys.maxsize)


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
             height=2276,
             width=4552,
             fov=90):
  sim = Refer360Simulator(cache_root,
                          output_image_shape=(image_h, image_w),
                          height=height,
                          width=width,
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

    n_fovs = int((360 / args.angle_inc)*math.ceil(150/args.angle_inc))
    for image_feature_type in sorted(args.refer360_image_feature_type):
      if 'none' in image_feature_type:
        feats.append(NoImageFeatures())
      if 'random' in image_feature_type:
        feats.append(RandImageFeatures())
      if 'mean_pooled' in image_feature_type:
        feats.append(MeanPooledImageFeatures(cache_root=args.cache_root,
                                             image_list_file=args.image_list_file,
                                             n_fovs=n_fovs,
                                             feature_model=args.refer360_image_feature_model,
                                             no_lookahead=args.no_lookahead))
      if 'butd' in image_feature_type:
        feats.append(BUTDImageFeatures(cache_root=args.cache_root,
                                       image_list_file=args.image_list_file,
                                       butd_filename=args.butd_filename,
                                       n_fovs=n_fovs,
                                       no_lookahead=args.no_lookahead,
                                       use_object_embeddings=args.use_object_embeddings,
                                       center_model=args.refer360_center_model))
      if 'prior' in image_feature_type:
        for prior_method in args.refer360_prior_method.split(','):
          feats.append(PriorImageFeatures(
              prior_prefix=args.prior_prefix,
              prior_method=prior_method))
      assert len(feats) >= 1, 'len(feats) >= 1, {} >= 0'.format(len(feats))
    return feats

  @staticmethod
  def add_args(argument_parser):
    argument_parser.add_argument('--refer360_image_feature_type', nargs='+',
                                 choices=['none', 'random',
                                          'mean_pooled', 'butd', 'prior'],
                                 default=['mean_pooled'])
    argument_parser.add_argument('--refer360_image_feature_model',
                                 choices=['resnet', 'clip'],
                                 default='clip')
    argument_parser.add_argument('--refer360_center_model',
                                 choices=['resnet', ''],
                                 default='')
    argument_parser.add_argument('--refer360_prior_method',
                                 default='')

  def get_name(self):
    raise NotImplementedError('base class does not have get_name')

  def batch_features(self, feature_list):
    features = np.stack(feature_list)
    return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

  def get_features(self, state):
    raise NotImplementedError('base class does not have get_features')


class RandImageFeatures(Refer360ImageFeatures):
  feature_dim = 24

  def __init__(self):
    print('Random image features will be provided')
    self.features = np.random.randn(
        Refer360ImageFeatures.NUM_VIEWS, self.feature_dim)

  def get_features(self, state):
    return self.features

  def get_name(self):
    return 'random'


class NoImageFeatures(Refer360ImageFeatures):
  feature_dim = Refer360ImageFeatures.MEAN_POOLED_DIM

  def __init__(self):
    print('Image features not provided')
    self.features = np.zeros(
        (Refer360ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)

  def get_features(self, state):
    return self.features

  def get_name(self):
    return 'none'


class MeanPooledImageFeatures(Refer360ImageFeatures):
  def __init__(self,
               cache_root='',
               image_list_file='',
               feature_model='resnet',
               n_fovs=240,
               no_lookahead=False):
    self.feature_dim = MODEL2FEATURE_DIM[feature_model]
    self.feature_model = feature_model
    self.feature_prefix = MODEL2PREFIX[feature_model]
    self.features = {}
    self.n_fovs = n_fovs
    self.no_lookahead = no_lookahead

    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    nodes = meta['nodes']

    print('loading image features for refer360 from', image_list_file)
    print('no_lookahead:', self.no_lookahead)
    image_list = [line.strip()
                  for line in open(image_list_file)]

    fov2feat = load_cnn_features(
        image_list, cache_root, self.feature_prefix, self.n_fovs)
    pbar = tqdm(image_list)

    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]
      for idx in range(self.n_fovs):
        pano_fov = self._make_id(pano, idx)
        feats = np.zeros((Refer360ImageFeatures.NUM_VIEWS,
                          self.feature_dim), dtype=np.float32)
        feats[4, :] = fov2feat[pano_fov]

        if not self.no_lookahead:
          for neighbor in nodes[idx]['neighbor2dir']:
            n_direction = nodes[idx]['neighbor2dir'][neighbor]
            dir_idx = DIR2IDX[n_direction]
            neighbor_fov = self._make_id(pano, neighbor)
            feats[dir_idx, :] = fov2feat[neighbor_fov]
        self.features[pano_fov] = feats
    del fov2feat

  def _make_id(self, scanId, viewpointId):
    return '{}_{}'.format(scanId, viewpointId)

  def get_features(self, state):
    long_id = self._make_id(state.scanId, state.viewpointId)
    # Return feature of all the 36 views
    return self.features[long_id]

  def get_name(self):
    name = 'mean_pooled.'+self.feature_model
    if self.no_lookahead:
      name += 'NOLA'
    return name


class BUTDImageFeatures(Refer360ImageFeatures):
  def __init__(self, butd_filename='',
               cache_root='',
               image_list_file='',
               n_fovs=240,
               no_lookahead=False,
               center_model='',
               use_object_embeddings=False,
               word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
               obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json'):

    self.use_object_embeddings = use_object_embeddings
    if self.use_object_embeddings:
      self.objemb, self.feature_dim = True, 300

      vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
          obj_dict_file, return_all=True)
      self.name2vg, self.vg2name = name2vg, vg2name

      print('Loading word embeddings...')
      self.w2v = load_vectors(word_embedding_path, self.name2vg)
      missing = []
      for w in self.name2vg:
        if w not in self.w2v:
          missing += [w]
      print('Missing object names:', ' '.join(missing))
    else:
      self.objemb, self.feature_dim = False, 2048
      self.w2v, self.vg2name, self.name2vg = None, None, None

    self.features = defaultdict(list)

    self.n_fovs = n_fovs
    self.no_lookahead = no_lookahead
    self.features = {}
    print('Loading bottom-up top-down features')
    print('no_lookahead:', self.no_lookahead)

    meta_file = os.path.join(cache_root, 'meta.npy')
    meta = np.load(meta_file, allow_pickle=True)[()]
    nodes = meta['nodes']

    image_list = [line.strip()
                  for line in open(image_list_file)]
    self.center_model = center_model
    if self.center_model:
      if self.center_model not in ['resnet']:
        raise NotImplementedError(
            '{} not implemented or not compatible'.format(center_model))
      center_prefix = MODEL2PREFIX[self.center_model]
      print('loading center model...')
      fov2cnn = load_cnn_features(
          image_list, cache_root, center_prefix, n_fovs)
    else:
      fov2cnn = {}

    print('loading BUTD features...', image_list_file)
    fov2feat = load_butd(butd_filename,
                         w2v=self.w2v,
                         vg2name=self.vg2name,
                         keys=['features'])['features']
    print('loaded BUTD features!', image_list_file)
    print('loading image features for refer360...')

    pbar = tqdm(image_list)
    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]

      for idx in range(self.n_fovs):
        feats = np.zeros(
            (Refer360ImageFeatures.NUM_VIEWS, self.feature_dim), dtype=np.float32)

        if self.center_model:
          feats[4, :] = fov2cnn['{}_{}'.format(pano, idx)]
        else:
          feats[4, :] = fov2feat['{}_{}'.format(pano, idx)]

        if not self.no_lookahead:
          for neighbor in nodes[idx]['neighbor2dir']:
            n_direction = nodes[idx]['neighbor2dir'][neighbor]
            dir_idx = DIR2IDX[n_direction]
            feats[dir_idx, :] = fov2feat['{}_{}'.format(pano, neighbor)]
        pano_fov = self._make_id(pano, idx)
        self.features[pano_fov] = feats
    print('image features for refer360 are prepared!')
    del fov2feat
    del fov2cnn

  def _make_id(self, scanId, viewpointId):
    return '{}_{}'.format(scanId, viewpointId)

  def get_features(self, state):
    long_id = self._make_id(state.scanId, state.viewpointId)
    return self.features[long_id]

  def get_name(self):
    name = 'butd'
    if self.use_object_embeddings:
      name += '_oe'
    if self.center_model:
      name += 'CM'+self.center_model
    if self.no_lookahead:
      name += 'NOLA'
    return name


class PriorImageFeatures(Refer360ImageFeatures):
  def __init__(self,
               prior_prefix='img_features/refer360_30degrees_',
               prior_method='vg'):

    self.feature_dim = 300
    self.features = {}
    self.prior_method = prior_method

    FIELDNAMES = ['pano_fov', 'features']
    decode_config = [
        ('features', (9, 300), np.float32),
    ]
    prior_file = prior_prefix + '{}.tsv'.format(prior_method)

    print('loading prior image features from', prior_file)
    with open(prior_file) as f:
      reader = csv.DictReader(f, FIELDNAMES, delimiter='\t')
      for i, item in enumerate(reader):
        for key, shape, dtype in decode_config:
          item[key] = np.frombuffer(
              base64.decodebytes(item[key].encode()), dtype=dtype).reshape(9, 300)
        pano_fov = item['pano_fov']
        features = item['features']
        self.features[pano_fov] = features
    print('image features for refer360 are prepared!')

  def _make_id(self, scanId, viewpointId):
    return '{}_{}'.format(scanId, viewpointId)

  def get_features(self, state):
    long_id = self._make_id(state.scanId, state.viewpointId)
    return self.features[long_id]

  def get_name(self):
    name = '_prior{}NOLA'.format(self.prior_method)
    return name


class Refer360EnvBatch():
  ''' A simple wrapper for a batch of MatterSim environments,
      using discretized viewpoints and pretrained features '''

  def __init__(self, batch_size, beam_size,
               cache_root='',
               height=2276,
               width=4552,
               args=None):
    self.sims = []
    self.batch_size = batch_size
    self.beam_size = beam_size
    self.cache_root = cache_root
    if args:
      if args.prefix == 'refer360':
        width, height = 4552, 2276
      elif args.prefix in ['touchdown', 'td']:
        width, height = 3000, 1500
      else:
        raise NotImplementedError()
    sim = make_sim(cache_root,
                   image_w=Refer360ImageFeatures.IMAGE_W,
                   image_h=Refer360ImageFeatures.IMAGE_H,
                   fov=Refer360ImageFeatures.VFOV,
                   height=height,
                   width=width)

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
                'train': 'train',
                'dev': 'dev',
                'test': 'test'
                }


def load_datasets(splits,
                  root='',
                  use_intermediate=True):
  d = []
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
        instructions = ' '.join(refexp)
        new_datum['instructions'] = [instructions]
        new_datum['path'] = datum['intermediate_paths'][kk]
        new_datum['gt_actions_path'] = datum['intermediate_paths'][kk]
        if len(new_datum['path']) <= 1:
          continue
        act_length += [len(new_datum['path'])]
        sen_length += [len(new_datum['instructions'][0].split(' '))]

        converted.append(new_datum)
    else:
      instructions = ' '.join([' '.join(refexp)
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
    refer360_data = args.refer360_data
    cache_root = args.cache_root
    use_intermediate = args.use_intermediate
    use_gt_actions = args.use_gt_actions
    use_visited_embeddings = args.use_visited_embeddings
    use_oracle_embeddings = args.use_oracle_embeddings
    angle_inc = args.angle_inc

    self.deaf = args.deaf
    self.blind = args.blind
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
                                  root=refer360_data,
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
          self.language = language
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
                       cache_root=cache_root,
                       args=args)
    self.print_progress = False
    print('Refer360Batch loaded with %d instructions, using splits: %s' %
          (len(self.data), ','.join(splits)))
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
    if self.blind:
      vle = self._static_loc_embeddings[0]
      self._static_loc_embeddings = [
          np.zeros_like(vle) for viewIndex in range(9)]

  def set_beam_size(self, beam_size,
                    force_reload=False,
                    cache_root='',
                    args=None):
    # warning: this will invalidate the environment, self.reset() should be called afterward!
    try:
      invalid = (beam_size != self.beam_size)
    except:
      invalid = True
    if force_reload or invalid:
      self.beam_size = beam_size
      self.env = Refer360EnvBatch(self.batch_size, beam_size,
                                  cache_root=cache_root,
                                  args=args)

  def _gt_action(self, state, adj_loc_list, goalViewpointId, gt_path):
    '''
    Determine next action on the grount-truth path to goal,
    for supervised training.
    '''
    if len(gt_path) == 1:
      return 0, gt_path
    assert state.viewpointId == gt_path[0], 'state.viewpointId != gt_path[0] {} != {} {} {} {}'.format(
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
          visited_embedding = build_visited_embedding(
              adj_loc_list, item['visited_viewpoints'])
          action_embedding = np.concatenate(
              (action_embedding, visited_embedding), axis=-1)
        if self.use_oracle_embeddings:
          oracle_embedding = build_oracle_embedding(
              adj_loc_list, teacher_action)
          action_embedding = np.concatenate(
              (action_embedding, oracle_embedding), axis=-1)

        instructions = '. . .' if self.deaf else item['instructions']
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
            'instructions': instructions,
        }
        if include_teacher and self.notTest:
          ob['teacher'] = teacher_action
          self.batch[i]['gt_actions_path'] = new_path
          ob['deviation'] = self._deviation(state, item['path'])
          ob['progress'] = self._progress(state, item['path']),
          ob['distance'] = self._distance(state, item['path']),
        if self.deaf:
          ob['instr_encoding'] = np.array([1])
          ob['instr_length'] = 1
        else:
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
