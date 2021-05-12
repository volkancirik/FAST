''' Evaluation of agent trajectories '''
import os
import json
from collections import defaultdict

import numpy as np
import random
import pprint
pp = pprint.PrettyPrinter(indent=4)  # NoQA
from pprint import pprint

from refer360_env import Refer360Batch, Refer360ImageFeatures, load_datasets
from refer360_env import make_sim
import utils

from follower import BaseAgent
#import train

from collections import namedtuple


EvalResult = namedtuple(
    'EvalResult', 'nav_error, oracle_error, trajectory_steps, '
    'trajectory_length, success, oracle_success, spl, '
    'fov_accuracy, acc_20, acc_40, acc_60,'
    'cls, ndtw')


class Refer360Evaluation(object):
  ''' Results submission format:
      [{'instr_id': string,
        'trajectory':[(viewpoint_id, heading_rads, elevation_rads),]}] '''

  def __init__(self, splits,
               args=None,
               sim=None):

    prefix = args.prefix
    refer360_data = args.refer360_data

    error_margin = args.error_margin
    self.sim = sim

    self.splits = splits
    self.gt = {}
    self.instr_ids = []
    self.scans = []
    self.instructions = {}
    counts = defaultdict(int)
    refer360_data = load_datasets(splits, root=refer360_data)
    for item in refer360_data:
      path_id = item['path_id']
      count = counts[path_id]
      new_path_id = '{}*{}'.format(path_id, count)
      counts[path_id] += 1
      item['path_id'] = new_path_id
      item['gt_actions_path'] = item['path']
      self.gt[item['path_id']] = item
      self.scans.append(item['scan'])
      if prefix in ['refer360', 'touchdown']:
        self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in
                           range(len(item['instructions']))]
      else:
        raise NotImplementedError()
      for j, instruction in enumerate(item['instructions']):
        self.instructions['{}_{}'.format(item['path_id'], j)] = instruction
    self.scans = set(self.scans)
    self.instr_ids = set(self.instr_ids)

    self.nodes = self.sim.nodes
    self.distances = self.sim.distances
    self.error_margin = error_margin*270.0  # error_margin * max FOV distance

  def _get_nearest(self, scan, goal_id, path):
    near_id = path[0][0]
    near_d = self.distances[near_id][goal_id]
    for item in path:
      d = self.distances[item[0]][goal_id]
      if d < near_d:
        near_id = item[0]
        near_d = d
    return near_id

  # Metrics implemented here:
  # https://github.com/Sha-Lab/babywalk/blob/master/simulator/envs/env.py
  # L282 - L324

  def ndtw(self, prediction, reference):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction) + 1):
      for j in range(1, len(reference) + 1):
        best_previous_cost = min(dtw_matrix[i - 1][j],
                                 dtw_matrix[i][j - 1],
                                 dtw_matrix[i - 1][j - 1])
        cost = self.distances[prediction[i - 1]][reference[j - 1]]
        dtw_matrix[i][j] = cost + best_previous_cost
    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw / (self.error_margin * len(reference)))
    return ndtw

  def length(self, nodes):
    return float(np.sum([self.distances[edge[0]][edge[1]]
                         for edge in zip(nodes[:-1], nodes[1:])]))

  def cls(self, prediction, reference):
    coverage = np.mean([np.exp(
        -np.min([self.distances[u][v] for v in prediction]) / self.error_margin
    ) for u in reference])
    expected = coverage * self.length(reference)
    score = expected \
        / (expected + np.abs(expected - self.length(prediction)))
    return coverage * score

  def _score_item(self, instr_id, path):
    ''' Calculate error based on the final position in trajectory, and also
        the closest position (oracle stopping rule). '''

    key = instr_id.split('_')[0]
    if key in self.gt:
      gt = self.gt[key]
    elif int(key) in self.gt:
      gt = self.gt[int(key)]
    else:
      print('int({}) or str({}) not in self.gt'.format(key, key))
      quit(1)

    start = gt['path'][0]

    assert start == path[0][0], \
        'Result trajectories should include the start position'
    goal = gt['path'][-1]
    final_position = path[-1][0]

    self.sim.look_fov(final_position)
    res = self.sim.get_image_coordinate_for(gt['gt_lng'], gt['gt_lat'])
    metrics = defaultdict(float)
    if res != None:
      fov_accuracy = 1.0
      distance = np.sqrt(((res[0] - 200)
                          ** 2 + (res[1] - 200)**2))
      for th in [20, 40, 60]:
        metrics['acc_{}'.format(th)] += int(distance <= th)
    else:
      fov_accuracy = 0
      for th in [20, 40, 60]:
        metrics['acc_{}'.format(th)] += 0

    nearest_position = self._get_nearest(gt['scan'], goal, path)
    nav_error = self.distances[final_position][goal]
    oracle_error = self.distances[nearest_position][goal]
    trajectory_steps = len(path)-1
    trajectory_length = 0  # Work out the length of the path in meters
    prev = path[0]
    for curr in path[1:]:
      trajectory_length += self.distances[prev[0]][curr[0]]
      prev = curr

    success = nav_error < self.error_margin
    # check for type errors
    # assert success == True or success == False
    # check for type errors
    oracle_success = oracle_error < self.error_margin
    # assert oracle_success == True or oracle_success == False

    sp_length = 0
    prev = gt['path'][0]
    sp_length = self.distances[gt['path'][0]][gt['path'][-1]]

    traj_length = max(trajectory_length, sp_length)
    spl = 0.0 if nav_error >= self.error_margin or traj_length == 0 else \
        (float(sp_length) / traj_length)

    prediction_path = [p[0] for p in path]
    cls = self.cls(prediction_path, gt['path'])
    ndtw = self.ndtw(prediction_path, gt['path'])

    return EvalResult(nav_error=nav_error, oracle_error=oracle_error,
                      trajectory_steps=trajectory_steps,
                      trajectory_length=trajectory_length, success=success,
                      oracle_success=oracle_success,
                      spl=spl,
                      fov_accuracy=fov_accuracy,
                      acc_20=metrics['acc_20'],
                      acc_40=metrics['acc_40'],
                      acc_60=metrics['acc_60'],
                      cls=cls,
                      ndtw=ndtw)

  def score_results(self, results):
    # results should be a dictionary mapping instr_ids to dictionaries,
    # with each dictionary containing (at least) a 'trajectory' field
    # return a dict with key being a evaluation metric
    self.scores = defaultdict(list)

    loc2scores, scene2scores, traj_length2scores, inst_length2scores = {}, {}, {}, {}
    for key in EvalResult._fields:
      loc2scores[key] = defaultdict(list)
      scene2scores[key] = defaultdict(list)
      traj_length2scores[key] = defaultdict(list)
      inst_length2scores[key] = defaultdict(list)

    model_scores = []
    instr_ids = set(self.instr_ids)

    instr_count = 0
    done = list()
    for instr_id, result in results.items():

      if instr_id in instr_ids:
        instr_count += 1
        instr_ids.remove(instr_id)
        eval_result = self._score_item(instr_id, result['trajectory'])

        img_path = self.gt[result['instr_id'].split('_')[0]]['img_src']
        img_loc = img_path.split('/')[3]
        img_scene = img_path.split('/')[4]
        traj_length = len(
            self.gt[result['instr_id'].split('_')[0]]['gt_actions_path'])

        if 'instr_encoding' in result:
          inst_length = result['instr_encoding'].shape[0]
        else:
          inst_length = 0

        self.scores['nav_error'].append(eval_result.nav_error)
        self.scores['oracle_error'].append(eval_result.oracle_error)
        self.scores['trajectory_steps'].append(
            eval_result.trajectory_steps)
        self.scores['trajectory_length'].append(
            eval_result.trajectory_length)
        self.scores['success'].append(eval_result.success)
        self.scores['oracle_success'].append(
            eval_result.oracle_success)
        self.scores['spl'].append(eval_result.spl)
        self.scores['fov_accuracy'].append(eval_result.fov_accuracy)
        self.scores['acc_20'].append(eval_result.acc_20)
        self.scores['acc_40'].append(eval_result.acc_40)
        self.scores['acc_60'].append(eval_result.acc_60)
        self.scores['cls'].append(eval_result.cls)
        self.scores['ndtw'].append(eval_result.ndtw)

        for sk in self.scores.keys():
          loc2scores[sk][img_loc].append(self.scores[sk][-1])
          scene2scores[sk][img_scene].append(self.scores[sk][-1])
          traj_length2scores[sk][traj_length].append(self.scores[sk][-1])
          inst_length2scores[sk][inst_length].append(self.scores[sk][-1])

        done.append((instr_id, instr_count))
        if 'score' in result:
          model_scores.append(result['score'].item())

    rand_idx = random.choice(range(instr_count))
    rand_result = results[done[rand_idx][0]]
    key = done[rand_idx][0].split('_')[0]
    if key in self.gt:
      gt = self.gt[key]
    elif int(key) in self.gt:
      gt = self.gt[int(key)]

    print('\n>>>>pred', [p[0] for p in rand_result['trajectory']])
    print('\n>>>>gt', gt['path'])
    print('\n>>>>gt-path', gt['gt_path'])
    print('\n>>>>gt-annotationid', gt['annotationid'])
    print('\n>>>>gt-actionid', gt['actionid'])
    print('\n>>>>gt-instructions', gt['instructions'])

    assert len(instr_ids) == 0, \
        'Missing %d of %d instruction ids from %s' % (
            len(instr_ids), len(self.instr_ids), ','.join(self.splits))

    assert len(self.scores['nav_error']) == len(self.instr_ids)
    score_summary = {
        'nav_error': np.average(self.scores['nav_error']),
        'oracle_error': np.average(self.scores['oracle_error']),
        'steps': np.average(self.scores['trajectory_steps']),
        'lengths': np.average(self.scores['trajectory_length']),
        'success': float(
            sum(self.scores['success']) / len(self.scores['success'])),
        'oracle_rate': float(sum(self.scores['oracle_success'])
                             / len(self.scores['oracle_success'])),
        'spl': float(sum(self.scores['spl'])) / len(self.scores['spl']),
        'fov_accuracy': float(
            sum(self.scores['fov_accuracy']) / len(self.scores['fov_accuracy'])),
        'acc_20': float(
            sum(self.scores['acc_20']) / len(self.scores['acc_20'])),
        'acc_40': float(
            sum(self.scores['acc_40']) / len(self.scores['acc_40'])),
        'acc_60': float(
            sum(self.scores['acc_60']) / len(self.scores['acc_60'])),
        'cls': float(sum(self.scores['cls'])) / len(self.scores['cls']),
        'ndtw': float(sum(self.scores['ndtw'])) / len(self.scores['ndtw']),
    }
    if len(model_scores) > 0:
      assert len(model_scores) == instr_count
      score_summary['model_score'] = np.average(model_scores)

    num_successes = len(
        [i for i in self.scores['nav_error'] if i < self.error_margin])
    # score_summary['success'] = float(num_successes)/float(len(self.scores['nav_error']))  # NoQA
    assert float(num_successes) / float(len(self.scores['nav_error'])) == score_summary['success']  # NoQA
    oracle_successes = len(
        [i for i in self.scores['oracle_error'] if i < self.error_margin])
    assert float(oracle_successes) / float(len(self.scores['oracle_error'])) == score_summary['oracle_rate']  # NoQA
    # score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_error']))  # NoQA

    analysis = {'loc2scores': loc2scores,
                'scene2scores': scene2scores,
                'traj_length2scores': traj_length2scores,
                'inst_length2scores': inst_length2scores
                }
    return score_summary, self.scores, analysis

  def score_file(self, output_file):
    ''' Evaluate each agent trajectory based on how close it got to the
    goal location '''
    with open(output_file) as f:
      return self.score_results(json.load(f))

  def score_test_file(self, output_file):
    with open(output_file) as f:
      _d = json.load(f)
      results = {}
      for item in _d:
        results[item['instr_id']] = item
      return self.score_results(results)

  def _path_segments(self, path):
    segs = []
    for i in range(len(path)-1):
      a = min(path[i], path[i+1])
      b = max(path[i], path[i+1])
      segs.append((a, b))
    return set(segs)

  def _inspect(self, instr_id, traj, eval_result):
    results = {'instr_id': instr_id}

    full_path = [p[0] for p in traj]
    path = [full_path[0]]
    for _vpt in full_path:
      if _vpt != path[-1]:
        path.append(_vpt)
    results['path'] = path

    plen = len(path)
    gt = self.gt[int(instr_id.split('_')[0])]
    gt_path = gt['path']
    glen = len(gt_path)

    # 1. No.X starts deviation
    _diff = 0
    while path[_diff] == gt_path[_diff]:
      _diff += 1
      if _diff == plen or _diff == glen:
        break
    if _diff == plen and _diff == glen:
      _diff = -1  # mark 'no deviation'
    results['ontrack'] = _diff

    # 2. Percentage starts deviation
    _percent_diff = float(_diff) / plen
    results['%ontrack'] = _percent_diff

    # 3. # of path segments on gt_path
    psegs = self._path_segments(path)
    gsegs = self._path_segments(gt_path)
    _shared_segs = len(psegs & gsegs)
    results['good_segments'] = _shared_segs

    # 4. % of gt_segment in rollout
    _s_r = float(_shared_segs) / len(psegs)
    results['good/rollout'] = _s_r

    # 5. % of gt_segment got covered
    _s_g = float(_shared_segs) / len(gsegs)
    results['good/gt'] = _s_g

    results['success'] = eval_result.success

    return results

  def inspect_results(self, results):
    inspection = defaultdict(list)
    evals = []
    instr_ids = set(self.instr_ids)
    instr_count = 0
    skipped_count = 0

    if type(results) is list:
      _res = results
      results = {}
      for item in _res:
        results[item['instr_id']] = item

    for instr_id, result in results.items():
      if instr_id in instr_ids:
        instr_count += 1
        instr_ids.remove(instr_id)
        eval_result = self._score_item(instr_id, result['trajectory'])
        evals.append(eval_result)
        res = self._inspect(instr_id, result['trajectory'], eval_result)
        for k, v in res.items():
          inspection[k].append(v)
      else:
        skipped_count += 1
    print('Inspected', instr_count)
    print('Skipped', skipped_count)
    return inspection, evals


def eval_simple_agents(args):
  ''' Run simple baselines on each split. '''
  img_features = Refer360ImageFeatures.from_args(args)

  sim = make_sim(args.cache_root,
                 Refer360ImageFeatures.IMAGE_W,
                 Refer360ImageFeatures.IMAGE_H,
                 Refer360ImageFeatures.VFOV)
  sim.load_maps()

  # TODO add touchdown
  if args.prefix == 'refer360':
    splits = ['val_seen',
              'val_unseen',
              'test_unseen',
              'test_seen']
  elif args.prefix == 'touchdown':
    splits = ['dev', 'test']
  else:
    raise NotImplementedError()
  for split in splits:
    env = Refer360Batch(img_features,
                        splits=[split],
                        args=args)
    ev = Refer360Evaluation([split],
                            args=args,
                            sim=sim)

    for agent_type in ['Stop', 'Shortest', 'Random']:
      outfile = '%s%s_%s_agent.json' % (
          args.RESULT_DIR, split, agent_type.lower())
      agent = BaseAgent.get_agent(agent_type)(env, outfile)
      agent.test()
      agent.write_results()
      score_summary, _, _ = ev.score_file(outfile)
      print('\n%s' % agent_type)
      pp.pprint(score_summary)


def eval_seq2seq(args):
  ''' Eval sequence to sequence models on val splits (iteration selected from
  training error) '''
  outfiles = [
      args.RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
      args.RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
  ]
  sim = make_sim(args.cache_root,
                 Refer360ImageFeatures.IMAGE_W,
                 Refer360ImageFeatures.IMAGE_H,
                 Refer360ImageFeatures.VFOV)
  sim.load_maps()

  for outfile in outfiles:
    for split in ['val_seen', 'val_unseen']:
      ev = Refer360Evaluation([split],
                              args=args,
                              sim=sim)
      score_summary, _, _ = ev.score_file(outfile % split)
      print('\n%s' % outfile)
      pp.pprint(score_summary)


def eval_outfiles(args):
  outfolder = args.results_path
  splits = ['val_seen', 'val_unseen']
  sim = make_sim(args.cache_root,
                 Refer360ImageFeatures.IMAGE_W,
                 Refer360ImageFeatures.IMAGE_H,
                 Refer360ImageFeatures.VFOV)
  sim.load_maps()

  for _f in os.listdir(outfolder):
    outfile = os.path.join(outfolder, _f)
    _splits = []
    for s in splits:
      if s in outfile:
        _splits.append(s)
    ev = Refer360Evaluation(_splits,
                            args=args,
                            sim=sim)
    score_summary, _, _ = ev.score_file(outfile)
    print('\n', outfile)
    pp.pprint(score_summary)


if __name__ == '__main__':
  from train import make_arg_parser
  # TODO: take function to run as argument
  # parser = make_arg_parser()
  # parser.add_argument('--results_path', type=str,
  #                     default='')

  # utils.run(parser, eval_outfiles)
  utils.run(make_arg_parser(), eval_simple_agents)
