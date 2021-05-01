import torch
from torch import optim
import json
import os
import os.path
import sys
import time
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utils import colorize, filter_param
from utils import get_confusion_matrix_image, get_bar_image

from model import BertEncoder, EncoderLSTM, AttnDecoderLSTM
from model import CogroundDecoderLSTM, ProgressMonitor, DeviationMonitor
from model import SpeakerEncoderLSTM, DotScorer, BacktrackButton
from follower import Seq2SeqAgent, RandomAgent
from scorer import Scorer

from env import R2RBatch, ImageFeatures
from refer360_env import Refer360Batch, Refer360ImageFeatures, make_sim

import eval
import refer360_eval
from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB
from tensorboardX import SummaryWriter


def get_model_prefix(args, image_feature_list,
                     dump_args=False):
  image_feature_name = '+'.join(
      [featurizer.get_name() for featurizer in image_feature_list])
  nn = ('{}{}{}{}{}{}{}{}{}'.format(
      ('_bt' if args.bert else ''),
      ('_sc' if args.scorer else ''),
      ('_mh' if args.num_head > 1 else ''),
      ('_cg' if args.coground else ''),
      ('_pm' if args.prog_monitor else ''),
      ('_sa' if args.soft_align else ''),
      ('_bi' if args.bidirectional else ''),
      ('_gl' if args.use_glove else ''),
      ('_ve' if args.use_visited_embeddings else ''),
  ))
  model_prefix = 'follower{}_F{}_IMGF{}_NHe{}_Hid{}_ENL{}_DR{}'.format(
      nn,
      args.feedback_method,
      image_feature_name,
      args.num_head,
      args.hidden_size,
      args.encoder_num_layers,
      args.dropout_ratio)
  if args.use_train_subset:
    model_prefix = 'trainsub_' + model_prefix
  if args.use_pretraining:
    model_prefix = model_prefix.replace(
        'follower', 'follower_with_pretraining', 1)

  if dump_args:
    now = datetime.datetime.now()
    args_fn = '%s.args-%d-%d-%d,%d:%d:%d' % (model_prefix, now.year, now.month,
                                             now.day, now.hour, now.minute, now.second)
    with open(os.path.join(args.PLOT_DIR, args_fn), 'w') as out_file:
      out_file.write(' '.join(sys.argv))
      out_file.write('\n')
      json.dump(dict(args), out_file)
      out_file.write('\n')
  return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
  agent.results_path = results_path
  agent.test(
      use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


def train(args, train_env, agent, optimizers, n_iters, val_envs=None):
  ''' Train on training set, validating on both seen and unseen. '''

  if val_envs is None:
    val_envs = {}
  split_string = '-'.join(train_env.splits)
  print('Training with %s feedback' % args.feedback_method)
  writer_path = os.path.join(args.PLOT_DIR, get_model_prefix(
      args, train_env.image_features_list, dump_args=True))
  writer = SummaryWriter(writer_path)
  print('tensorboard path is', writer_path)

  data_log = defaultdict(list)
  start = time.time()

  def make_path(n_iter):
    return os.path.join(
        args.SNAPSHOT_DIR, '%s_%s_iter_%d' % (
            get_model_prefix(args, train_env.image_features_list),
            split_string, n_iter))

  best_metrics = {}
  last_model_saved = {}
  for idx in range(0, n_iters, args.log_every):
    agent.env = train_env

    interval = min(args.log_every, n_iters-idx)
    iter = idx + interval
    data_log['iteration'].append(iter)
    loss_str = ''

    # Train for log_every interval
    env_name = 'train'
    agent.train(optimizers, interval, feedback=args.feedback_method)
    _loss_str, losses, images = agent.get_loss_info()
    loss_str += env_name + ' ' + _loss_str
    for k, v in losses.items():
      data_log['%s %s' % (env_name, k)].append(v)
      writer.add_scalar('{} {}'.format(env_name, k), v, iter)

    for k, v in images.items():
      img_conf = get_confusion_matrix_image(
          [[str(v) for v in range(v.size(1))], [str(v) for v in range(v.size(0))]], v.cpu().numpy(), '')
      writer.add_image(k, img_conf, iter)

    save_log = []
    # Run validation
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
      agent.env = val_env
      # Get validation loss under the same conditions as training
      agent.test(use_dropout=True, feedback=args.feedback_method,
                 allow_cheat=True)
      _loss_str, losses, _ = agent.get_loss_info()
      loss_str += ', ' + env_name + ' ' + _loss_str
      for k, v in losses.items():
        data_log['%s %s' % (env_name, k)].append(v)
        writer.add_scalar('{} {}'.format(env_name, k), v, iter)

      agent.results_path = '%s/%s_%s_iter_%d.json' % (
          args.RESULT_DIR, get_model_prefix(
              args, train_env.image_features_list),
          env_name, iter)

      # Get validation distance from goal under evaluation conditions
      agent.test(use_dropout=False, feedback='argmax')

      print('evaluating on {}'.format(env_name))
      score_summary, all_scores, score_analysis = evaluator.score_results(
          agent.results)

      scores_path = make_path(iter) + '_%s_scores.npy' % (
          env_name)
      print('scores stats are dumped to %s' % scores_path)
      with open(scores_path, 'wb') as f:
        np.save(f, all_scores)

      for metric, val in sorted(score_summary.items()):

        writer.add_scalar('{} {}'.format(env_name, metric), val, iter)
        data_log['%s %s' % (env_name, metric)].append(val)
        if metric in args.metrics.split(','):

          for analysis in score_analysis:
            keys = sorted(score_analysis[analysis][metric].keys())
            means = [np.mean(score_analysis[analysis][metric][key])
                     for key in keys]
            stds = [np.std(score_analysis[analysis][metric][key])
                    for key in keys]
            x_pos = np.arange(len(keys))
            img_bar = get_bar_image(x_pos, keys, means, stds)
            writer.add_image(analysis+'_'+metric, img_bar, iter)

          loss_str += ', %s: %.3f' % (metric, val)
          key = (env_name, metric)
          if key not in best_metrics or best_metrics[key] < val:
            best_metrics[key] = val
            if not args.no_save:
              model_path = make_path(iter) + '_%s-%s=%.3f' % (
                  env_name, metric, val)
              save_log.append(
                  'new best, saved model to %s' % model_path)
              agent.save(model_path)
              agent.write_results()
              if key in last_model_saved:
                for old_model_path in last_model_saved[key]:
                  if os.path.isfile(old_model_path):
                    os.remove(old_model_path)
              # last_model_saved[key] = [agent.results_path] +\
              last_model_saved[key] = [] +\
                  list(agent.modules_paths(model_path))

    print(('%s (%d %d%%) %s' % (
        timeSince(start, float(iter)/n_iters),
        iter, float(iter)/n_iters*100, loss_str)))
    for s in save_log:
      print(colorize(s))

    if not args.no_save:
      if args.save_every and iter % args.save_every == 0:
        agent.save(make_path(iter))

      df = pd.DataFrame(data_log)
      df.set_index('iteration')
      df_path = '%s/%s_%s_log.csv' % (
          args.PLOT_DIR, get_model_prefix(
              args, train_env.image_features_list), split_string)
      print('data_log written to', df_path)
      df.to_csv(df_path)


def setup(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def make_more_train_env(args, train_vocab_path, train_splits):
  setup(args.seed)
  if args.env == 'r2r':
    EnvBatch = R2RBatch
    ImgFeatures = ImageFeatures
  elif args.env == 'refer360':
    EnvBatch = Refer360Batch
    ImgFeatures = Refer360ImageFeatures

  else:
    raise NotImplementedError(
        'this {} environment is not implemented.'.format(args.env))

  image_features_list = ImgFeatures.from_args(args)
  vocab = read_vocab(train_vocab_path, args.language)
  tok = Tokenizer(vocab=vocab)

  train_env = EnvBatch(image_features_list,
                       splits=train_splits,
                       tokenizer=tok,
                       args=args)
  return train_env


def make_scorer(args,
                action_embedding_size=-1,
                feature_size=-1):
  bidirectional = args.bidirectional

  enc_hidden_size = int(
      args.hidden_size/2) if bidirectional else args.hidden_size
  traj_encoder = try_cuda(SpeakerEncoderLSTM(action_embedding_size, feature_size,
                                             enc_hidden_size, args.dropout_ratio,
                                             bidirectional=args.bidirectional))
  scorer_module = try_cuda(DotScorer(enc_hidden_size, enc_hidden_size))
  scorer = Scorer(scorer_module, traj_encoder)
  if args.load_scorer != '':
    scorer.load(args.load_scorer)
    print(colorize('load scorer traj ' + args.load_scorer))
  elif args.load_traj_encoder != '':
    scorer.load_traj_encoder(args.load_traj_encoder)
    print(colorize('load traj encoder ' + args.load_traj_encoder))
  return scorer


def make_follower(args, vocab,
                  action_embedding_size=-1,
                  feature_size=-1):
  if args.random_baseline:
    print('using random agent')
    agent = RandomAgent
    return agent

  enc_hidden_size = int(
      args.hidden_size//2) if args.bidirectional else args.hidden_size
  glove = np.load(args.glove_path) if args.use_glove else None

  if args.bert:
    Encoder = BertEncoder
    args.hidden_size = 768
  else:
    Encoder = EncoderLSTM
  args.visual_hidden_size = args.hidden_size * 2

  Decoder = CogroundDecoderLSTM if args.coground else AttnDecoderLSTM
  word_embedding_size = int(
      args.hidden_size / 2) if args.coground or args.bidirectional else args.hidden_size
  encoder = try_cuda(Encoder(len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx, args.dropout_ratio,
                             bidirectional=args.bidirectional,
                             glove=glove,
                             num_layers=args.encoder_num_layers))

  decoder = try_cuda(Decoder(
      action_embedding_size, args.hidden_size, args.dropout_ratio,
      feature_size=feature_size,
      num_head=args.num_head,
      max_len=args.max_input_length,
      visual_hidden_size=args.visual_hidden_size))
  if not args.coground and args.use_visited_embeddings:
    action_embedding_size -= 64
  prog_monitor = try_cuda(ProgressMonitor(action_embedding_size,
                                          args.hidden_size, text_len=args.max_input_length)) if args.prog_monitor else None
  bt_button = try_cuda(BacktrackButton()) if args.bt_button else None
  dev_monitor = try_cuda(DeviationMonitor(action_embedding_size,
                                          args.hidden_size)) if args.dev_monitor else None

  agent = Seq2SeqAgent(
      None, '', encoder, decoder, args.max_episode_len,
      max_instruction_length=args.max_input_length,
      attn_only_verb=args.attn_only_verb,
      clip_rate=args.clip_rate)
  agent.prog_monitor = prog_monitor
  agent.dev_monitor = dev_monitor
  agent.bt_button = bt_button
  agent.soft_align = args.soft_align

  if args.scorer:
    agent.scorer = make_scorer(args,
                               action_embedding_size=action_embedding_size,
                               feature_size=feature_size)

  if args.load_follower != '':
    scorer_exists = os.path.isfile(args.load_follower + '_scorer_enc')
    agent.load(args.load_follower, load_scorer=(
        args.load_scorer == '' and scorer_exists))
    print(colorize('load follower ' + args.load_follower))

  return agent


def make_env_and_models(args, train_vocab_path, train_splits, test_splits):
  setup(args.seed)
  if args.env == 'r2r':
    EnvBatch = R2RBatch
    ImgFeatures = ImageFeatures
    Eval = eval.Evaluation
    env_sim = None
  elif args.env == 'refer360':
    EnvBatch = Refer360Batch
    ImgFeatures = Refer360ImageFeatures
    Eval = refer360_eval.Refer360Evaluation
    sim = make_sim(args.cache_root,
                   Refer360ImageFeatures.IMAGE_W,
                   Refer360ImageFeatures.IMAGE_H,
                   Refer360ImageFeatures.VFOV)
    sim.load_maps()
    env_sim = sim
  else:
    raise NotImplementedError(
        'this {} environment is not implemented.'.format(args.env))

  image_features_list = ImgFeatures.from_args(args)

  vocab = read_vocab(train_vocab_path, args.language)
  tok = Tokenizer(vocab=vocab)

  train_env = EnvBatch(image_features_list,
                       splits=train_splits,
                       tokenizer=tok,
                       args=args) if len(train_splits) > 0 else None
  test_envs = {
      split: (EnvBatch(image_features_list,
                       splits=[split],
                       tokenizer=tok,
                       args=args),
              Eval([split],
                   sim=env_sim,
                   args=args))
      for split in test_splits}

  feature_size = sum(
      [featurizer.feature_dim for featurizer in image_features_list]) + 128
  if args.use_visited_embeddings:
    feature_size += 64
  if args.use_oracle_embeddings:
    feature_size += 64
  agent = make_follower(args, vocab,
                        action_embedding_size=feature_size,
                        feature_size=feature_size)
  agent.env = train_env

  return train_env, test_envs, agent


def train_setup(args, train_splits=['train']):
  val_splits = ['val_seen', 'val_unseen']
  if args.use_test_set:
    val_splits = ['val_seen', 'val_unseen']
#    val_splits = ['test_seen','test_unseen']
  if args.debug:
    args.log_every = 3
    args.n_iters = 2
    args.image_feature_type = ['none']
    args.refer360_image_feature_type = ['none']

  vocab = TRAIN_VOCAB

  if args.use_train_subset:
    train_splits = ['sub_' + split for split in train_splits]
    val_splits = ['sub_' + split for split in val_splits]
    vocab = SUBTRAIN_VOCAB

  train_env, val_envs, agent = make_env_and_models(
      args, vocab, train_splits, val_splits)

  if args.use_pretraining:
    pretrain_splits = args.pretrain_splits
    assert len(pretrain_splits) > 0, \
        'must specify at least one pretrain split'
    pretrain_env = make_more_train_env(
        args, vocab, pretrain_splits)

  if args.use_pretraining:
    return agent, train_env, val_envs, pretrain_env
  else:
    return agent, train_env, val_envs

# Test set prediction will be handled separately
# def test_setup(args):
#     train_env, test_envs, encoder, decoder = make_env_and_models(
#         args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'])
#     agent = Seq2SeqAgent(
#         None, '', encoder, decoder, max_episode_len,
#         max_instruction_length=MAX_INPUT_LENGTH)
#     return agent, train_env, test_envs


def train_val(args):
  ''' Train on the training set, and validate on seen and unseen splits. '''
  if args.use_pretraining:
    agent, train_env, val_envs, pretrain_env = train_setup(args)
  else:
    agent, train_env, val_envs = train_setup(args)

  m_dict = {
      'follower': [agent.encoder, agent.decoder],
      'pm': [agent.prog_monitor],
      'follower+pm': [agent.encoder, agent.decoder, agent.prog_monitor],
      'all': agent.modules()
  }
  if agent.scorer:
    m_dict['scorer_all'] = agent.scorer.modules()
    m_dict['scorer_scorer'] = [agent.scorer.scorer]

  optimizers = [optim.Adam(filter_param(m),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay) for m in m_dict[args.grad] if len(filter_param(m))]

  if args.use_pretraining:
    train(args, pretrain_env, agent, optimizers,
          args.n_pretrain_iters, val_envs=val_envs)

  print('will use device:', torch.cuda.get_device_name(0))
  train(args, train_env, agent, optimizers,
        args.n_iters, val_envs=val_envs)

# TODO
# def test_submission(args):
#     ''' Train on combined training and validation sets, and generate test
#     submission. '''


def make_arg_parser():
  parser = argparse.ArgumentParser()

  ImageFeatures.add_args(parser)
  Refer360ImageFeatures.add_args(parser)

  parser.add_argument('--load_scorer', type=str, default='')
  parser.add_argument('--load_follower', type=str, default='')
  parser.add_argument('--load_traj_encoder', type=str, default='')
  parser.add_argument('--feedback_method',
                      choices=['sample', 'teacher', 'sample1step', 'sample2step', 'sample3step', 'teacher+sample', 'recover', 'argmax'], default='teacher')
  parser.add_argument('--debug', action='store_true')

  parser.add_argument('--bidirectional', action='store_true')
  parser.add_argument('--hidden_size', type=int, default=512)
  parser.add_argument('--encoder_num_layers', type=int, default=2)
  parser.add_argument('--learning_rate', type=float, default=0.0001)
  parser.add_argument('--clip_rate', type=float, default=0.)
  parser.add_argument('--weight_decay', type=float, default=0.0005)
  parser.add_argument('--dropout_ratio', type=float, default=0.5)
  parser.add_argument('--bert', action='store_true')
  parser.add_argument('--num_head', type=int, default=1)
  parser.add_argument('--scorer', action='store_true')
  parser.add_argument('--coground', action='store_false')
  parser.add_argument('--prog_monitor', action='store_false')
  parser.add_argument('--dev_monitor', action='store_true')
  parser.add_argument('--soft_align', action='store_true')
  parser.add_argument('--bt_button', action='store_true')
  parser.add_argument('--use_glove', action='store_true')
  parser.add_argument('--attn_only_verb', action='store_true')

  parser.add_argument('--use_gt_actions', action='store_true')
  parser.add_argument('--use_visited_embeddings', action='store_true')
  parser.add_argument('--use_oracle_embeddings', action='store_true')

  parser.add_argument('--n_iters', type=int, default=100000)
  parser.add_argument('--log_every', type=int, default=5000)
  parser.add_argument('--save_every', type=int, default=5000)
  parser.add_argument('--max_input_length', type=int, default=80)
  parser.add_argument('--max_episode_len', type=int, default=20)
  parser.add_argument('--grad', type=str, default='all')
  parser.add_argument('--metrics', type=str,
                      default='success',
                      help='Success metric, default=success')
  parser.add_argument('--use_pretraining', action='store_true')
  parser.add_argument('--pretrain_splits', nargs='+', default=[])
  parser.add_argument('--n_pretrain_iters', type=int, default=50000)
  parser.add_argument('--use_train_subset', action='store_true',
                      help='use a subset of the original train data for validation')
  parser.add_argument('--use_test_set', action='store_true')
  parser.add_argument('--no_save', action='store_true')
  parser.add_argument('--random_baseline', action='store_true')
  parser.add_argument('--seed', type=int, default=10)
  parser.add_argument('--beam_size', type=int, default=1)

  parser.add_argument('--prefix', type=str, default='R2R')
  parser.add_argument('--language', type=str, default='en-ALL')
  parser.add_argument('--glove_path', type=str,
                      default='tasks/R2R/data/train_glove.en-ALL.npy')
  parser.add_argument('--error_margin', type=float, default=3.0)
  parser.add_argument('--use_intermediate', action='store_true')
  parser.add_argument('--add_asterix', action='store_true')

  parser.add_argument('--refer360_root', type=str,
                      default='refer360_data')
  parser.add_argument('--angle_inc', type=int, default=30)

  parser.add_argument('--deaf', action='store_true')
  parser.add_argument('--blind', action='store_true')

  parser.add_argument('--verbose', action='store_true')

  return parser


if __name__ == '__main__':
  torch.backends.cudnn.enabled = False
  utils.run(make_arg_parser(), train_val)
