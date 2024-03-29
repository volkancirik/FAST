import torch
from torch import optim

import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab, Tokenizer, timeSince, try_cuda
from env import R2RBatch, ImageFeatures
from refer360_env import Refer360Batch, Refer360ImageFeatures

from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from speaker import Seq2SeqSpeaker
import eval_speaker
from train import get_word_embedding_size
from vocab import SUBTRAIN_VOCAB, TRAIN_VOCAB, TRAINVAL_VOCAB


def get_model_prefix(args, image_feature_list):
  image_feature_name = "+".join(
      [featurizer.get_name() for featurizer in image_feature_list])
  model_prefix = 'speaker_{}_{}'.format(
      args.feedback_method, image_feature_name)
  if args.use_train_subset:
    model_prefix = 'trainsub_' + model_prefix
  return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
  agent.results_path = results_path
  agent.test(
      use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


def filter_param(param_list):
  return [p for p in param_list if p.requires_grad]


def train(args, train_env, agent, val_envs=None):
  ''' Train on training set, validating on both seen and unseen. '''

  if val_envs is None:
    val_envs = {}

  print('Training with %s feedback' % args.feedback_method)
  encoder_optimizer = optim.Adam(
      filter_param(agent.encoder.parameters()),
      lr=args.learning_rate,
      weight_decay=args.weight_decay)
  decoder_optimizer = optim.Adam(
      filter_param(agent.decoder.parameters()),
      lr=args.learning_rate,
      weight_decay=args.weight_decay)

  data_log = defaultdict(list)
  start = time.time()

  split_string = "-".join(train_env.splits)

  def make_path(n_iter):
    return os.path.join(
        args.SNAPSHOT_DIR, '%s_%s_iter_%d' % (
            get_model_prefix(args, train_env.image_features_list),
            split_string, n_iter))

  best_metrics = {}
  last_model_saved = {}
  for idx in range(0, args.n_iters, args.log_every):
    agent.env = train_env

    interval = min(args.log_every, args.n_iters-idx)
    iter = idx + interval
    data_log['iteration'].append(iter)

    # Train for log_every interval
    agent.train(encoder_optimizer, decoder_optimizer, interval,
                feedback=args.feedback_method)
    train_losses = np.array(agent.losses)
    assert len(train_losses) == interval
    train_loss_avg = np.average(train_losses)
    data_log['train loss'].append(train_loss_avg)
    loss_str = 'train loss: %.4f' % train_loss_avg

    save_log = []
    # Run validation
    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
      agent.env = val_env
      # Get validation loss under the same conditions as training
      agent.test(use_dropout=True,
                 feedback=args.feedback_method,
                 allow_cheat=True)
      val_losses = np.array(agent.losses)
      val_loss_avg = np.average(val_losses)
      data_log['%s loss' % env_name].append(val_loss_avg)

      agent.results_path = '%s%s_%s_iter_%d.json' % (
          args.RESULT_DIR, get_model_prefix(
              args, train_env.image_features_list),
          env_name, iter)

      # Get validation distance from goal under evaluation conditions
      results = agent.test(use_dropout=False, feedback='argmax')
      if not args.no_save:
        agent.write_results()
      print("evaluating on {}".format(env_name))
      score_summary, _ = evaluator.score_results(results, verbose=True)

      loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
      for metric, val in score_summary.items():
        data_log['%s %s' % (env_name, metric)].append(val)
        if metric in ['bleu']:
          loss_str += ', %s: %.3f' % (metric, val)

          key = (env_name, metric)
          if key not in best_metrics or best_metrics[key] < val:
            best_metrics[key] = val
            if not args.no_save:
              model_path = make_path(iter) + "_%s-%s=%.3f" % (
                  env_name, metric, val)
              save_log.append(
                  "new best, saved model to %s" % model_path)
              agent.save(model_path)
              if key in last_model_saved:
                for old_model_path in \
                        agent._encoder_and_decoder_paths(
                            last_model_saved[key]):
                  os.remove(old_model_path)
              last_model_saved[key] = model_path

    print(('%s (%d %d%%) %s' % (
        timeSince(start, float(iter)/args.n_iters),
        iter, float(iter)/args.n_iters*100, loss_str)))
    for s in save_log:
      print(s)

    if not args.no_save:
      if args.save_every and iter % args.save_every == 0:
        agent.save(make_path(iter))

      df = pd.DataFrame(data_log)
      df.set_index('iteration')
      df_path = '%s%s_log.csv' % (
          args.PLOT_DIR, get_model_prefix(
              args, train_env.image_features_list))
      df.to_csv(df_path)


def setup():
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)


def make_speaker(args,
                 action_embedding_size=-1,
                 feature_size=-1):
  enc_hidden_size = args.hidden_size//2 if args.bidirectional else args.hidden_size
  wordvec = np.load(args.wordvec_path)

  vocab = read_vocab(TRAIN_VOCAB, args.language)
  word_embedding_size = get_word_embedding_size(args)
  encoder = try_cuda(SpeakerEncoderLSTM(
      action_embedding_size, feature_size, enc_hidden_size, args.dropout_ratio,
      bidirectional=args.bidirectional))
  decoder = try_cuda(SpeakerDecoderLSTM(
      len(vocab), word_embedding_size, args.hidden_size, args.dropout_ratio,
      wordvec=wordvec,
      wordvec_finetune=args.wordvec_finetune))
  agent = Seq2SeqSpeaker(
      None, "", encoder, decoder, args.max_input_length)
  return agent


def make_env_and_models(args, train_vocab_path, train_splits, test_splits,
                        test_instruction_limit=None,
                        instructions_per_path=None):
  setup()
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
  feature_size = sum(
      [featurizer.feature_dim for featurizer in image_features_list]) + 128
  if args.use_visited_embeddings:
    feature_size += 64
  if args.use_oracle_embeddings:
    feature_size += 64
  action_embedding_size = feature_size

  vocab = read_vocab(train_vocab_path, args.language)
  tok = Tokenizer(vocab=vocab)

  train_env = EnvBatch(image_features_list,
                       splits=train_splits,
                       tokenizer=tok,
                       args=args)

  enc_hidden_size = args.hidden_size//2 if args.bidirectional else args.hidden_size
  wordvec = np.load(args.wordvec_path)

  word_embedding_size = get_word_embedding_size(args)
  enc_hidden_size = 600  # refer360 >>>
  enc_hidden_size = 512  # refer360 >>>
  # enc_hidden_size = 512  # r2r >>>

  encoder = try_cuda(SpeakerEncoderLSTM(
      action_embedding_size, feature_size, enc_hidden_size, args.dropout_ratio,
      bidirectional=args.bidirectional))
  word_embedding_size = 300  # refer360 >>>>
  word_embedding_size = 300  # r2r >>>>
  hidden_size = 600  # refer360 >>>
  hidden_size = 512  # refer360 >>>
  # hidden_size = 512  # >>> r2r
  #hidden_size = args.hidden_size

  decoder = try_cuda(SpeakerDecoderLSTM(
      len(vocab), word_embedding_size, hidden_size, args.dropout_ratio,
      wordvec=wordvec,
      wordvec_finetune=args.wordvec_finetune))

  test_envs = {}
  for split in test_splits:
    b = EnvBatch(image_features_list,
                 splits=[split],
                 tokenizer=tok,
                 args=args)
    e = eval_speaker.SpeakerEvaluation(
        [split], instructions_per_path=instructions_per_path,
        args=args)
    test_envs[split] = (b, e)

  # TODO
  # test_envs = {
  #     split: (BatchEnv(image_features_list, batch_size=batch_size,
  #                      splits=[split], tokenizer=tok,
  #                      instruction_limit=test_instruction_limit,
  #                      prefix=args.prefix),
  #             eval_speaker.SpeakerEvaluation(
  #                 [split], instructions_per_path=instructions_per_path, ))
  #     for split in test_splits}

  return train_env, test_envs, encoder, decoder


def train_setup(args):
  train_splits = ['train']
  # val_splits = ['train_subset', 'val_seen', 'val_unseen']
  val_splits = ['val_seen', 'val_unseen']
  vocab = TRAIN_VOCAB

  if args.use_train_subset:
    train_splits = ['sub_' + split for split in train_splits]
    val_splits = ['sub_' + split for split in val_splits]
    vocab = SUBTRAIN_VOCAB

  instructions_per_path = None if args.prefix == 'R2R' else 1

  train_env, val_envs, encoder, decoder = make_env_and_models(
      args, vocab, train_splits, val_splits,
      instructions_per_path=instructions_per_path)
  agent = Seq2SeqSpeaker(
      train_env, "", encoder, decoder, instruction_len=args.max_input_length)
  return agent, train_env, val_envs


# TODO
# Test set prediction will be handled separately
# def test_setup(args):
#     train_env, test_envs, encoder, decoder = make_env_and_models(
#         args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'])
#     agent = Seq2SeqSpeaker(
#         None, "", encoder, decoder, MAX_INSTRUCTION_LENGTH,
#         max_episode_len=max_episode_len)
#     return agent, train_env, test_envs


def train_val(args):
  ''' Train on the training set, and validate on seen and unseen splits. '''
  agent, train_env, val_envs = train_setup(args)
  train(args, train_env, agent, val_envs=val_envs)


# Test set prediction will be handled separately
# def test_submission(args):
#     ''' Train on combined training and validation sets, and generate test
#     submission. '''
#     agent, train_env, test_envs = test_setup(args)
#     train(args, train_env, agent)
#
#     test_env = test_envs['test']
#     agent.env = test_env
#
#     agent.results_path = '%s%s_%s_iter_%d.json' % (
#         args.RESULT_DIR, get_model_prefix(args, train_env.image_features_list),
#         'test', n_iters)
#     agent.test(use_dropout=False, feedback='argmax')
#     if not args.no_save:
#         agent.write_results()


def make_arg_parser():
  parser = argparse.ArgumentParser()
  ImageFeatures.add_args(parser)
  Refer360ImageFeatures.add_args(parser)
  parser.add_argument(
      "--use_train_subset", action='store_true',
      help="use a subset of the original train data for validation")

  parser.add_argument("--bidirectional", action='store_true')
  parser.add_argument("--word_embedding_size", type=int, default=300)
  #parser.add_argument("--hidden_size", type=int, default=512)
  parser.add_argument("--hidden_size", type=int, default=256)
  parser.add_argument("--learning_rate", type=float, default=0.0001)
  parser.add_argument("--weight_decay", type=float, default=0.0005)
  parser.add_argument("--dropout_ratio", type=float, default=0.5)
  parser.add_argument("--feedback_method",
                      choices=['teacher',
                               'sample'],
                      default='teacher')

  parser.add_argument("--n_iters", type=int, default=100000)
  parser.add_argument("--log_every", type=int, default=5000)
  parser.add_argument("--save_every", type=int, default=5000)
  parser.add_argument("--max_input_length", type=int, default=80)
  parser.add_argument("--seed", type=int, default=10)
  parser.add_argument("--beam_size", type=int, default=1)
  parser.add_argument("--no_save", action='store_true')

  parser.add_argument("--prefix", type=str, default='R2R')
  parser.add_argument("--language", type=str, default='en-ALL')
  parser.add_argument('--wordvec_path', type=str,
                      default='tasks/R2R/data/train_glove')
  parser.add_argument('--wordvec_finetune', action='store_true')
  parser.add_argument("--error_margin", type=float, default=3.0)
  parser.add_argument("--use_intermediate", action='store_true')
  parser.add_argument('--use_reading', action='store_true')
  parser.add_argument('--use_raw', action='store_true')
  parser.add_argument("--add_asterix", action='store_true')

  #parser.add_argument("--env", type=str, default='r2r')

  parser.add_argument('--img_features_root', type=str,
                      default='./img_features')
  parser.add_argument('--cache_root', type=str,
                      default='/projects/vcirik/refer360/data/cached_data_15degrees/')
  parser.add_argument('--image_list_file', type=str,
                      default='/projects/vcirik/refer360/data/imagelist.txt')
  parser.add_argument('--refer360_root', type=str,
                      default='/projects/vcirik/refer360/data/continuous_grounding')
  parser.add_argument("--angle_inc", type=int, default=30)

  parser.add_argument('--use_gt_actions', action='store_true')
  parser.add_argument('--use_absolute_location_embeddings',
                      action='store_true')
  parser.add_argument('--use_stop_embeddings', action='store_true')
  parser.add_argument('--use_timestep_embeddings', action='store_true')
  parser.add_argument('--use_visited_embeddings',
                      type=str,
                      choices=['', 'ones', 'zeros', 'count', 'pe'],
                      default='')
  parser.add_argument('--use_oracle_embeddings', action='store_true')
  parser.add_argument(
      '--use_object_embeddings', action='store_true')

  parser.add_argument('--metrics', type=str,
                      default='success',
                      help='Success metric, default=success')
  parser.add_argument('--deaf', action='store_true')
  parser.add_argument('--blind', action='store_true')
  parser.add_argument('--no_lookahead', action='store_true')
  parser.add_argument('--nextstep', action='store_true')

  parser.add_argument("--verbose", action='store_true')
  return parser


if __name__ == "__main__":
  #utils.run(make_arg_parser(), train_val)
  from train import make_arg_parser as m_a_p
  utils.run(m_a_p(), train_val)
