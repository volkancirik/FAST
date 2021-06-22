import argparse
import math
import os
from pprint import pprint
from refer360_utils import test_get_nears
from refer360_utils import generate_baseline_cooccurrences
from refer360_utils import get_visualgenome_stats
from refer360_utils import get_spatialsense_stats
from refer360_utils import get_wordnet_stats
from refer360_utils import get_dataset_stats
from refer360_utils import evaluate_fov_caches
from refer360_utils import dump_fov_caches
from refer360_utils import dump_oracle_caches
from refer360_utils import dump_fov_stats


parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix",
                    choices=['td', 'refer360', 'r360tiny'],
                    default='refer360',
                    help='dataset prefix, default: refer360')
parser.add_argument('--angle_inc', type=int,
                    choices=[30, 45, 60],
                    default=30,
                    help='degrees between fovs, default=15')
parser.add_argument('--output_root', type=str,
                    default='./img_features',
                    help='Dump folder path like default=./img_features')
parser.add_argument('--diag_mode', type=int,
                    choices=[-1, 0, 1],
                    default=0,
                    help='diagonal of cooccurrence data, default=-1')
parser.add_argument('--img_features_root', type=str,
                    default='./img_features',
                    help='img_features_root, default="./img_features"')
parser.add_argument('--obj_dict_file', type=str,
                    default='./tasks/FAST/data/vg_object_dictionaries.all.json',
                    help='obj_dict_file, default=./tasks/FAST/data/vg_object_dictionaries.all.json')
parser.add_argument('--word_embedding_path', type=str,
                    default='./tasks/FAST/data/cc.en.300.vec',
                    help='word embedding path, default=./tasks/FAST/data/cc.en.300.vec')

parser.add_argument('--cooccurrence_path', type=str,
                    default='./cooccurrences',
                    help='cooccurrence files root, default="./cooccurrences"')
parser.add_argument('--stats_path', type=str,
                    default='./stats',
                    help='stats files root, default="./stats"')
parser.add_argument('--version', type=str,
                    default='v5',
                    help='version of prior methods, default="v5"')
parser.add_argument('--prefix_iou', type=int,  default=10,
                    help='prefix_iou, default=10')
parser.add_argument("--cooccurrence_method",
                    choices=['all', 'lm', 'language',
                             'vision', 'word_emb', 'dataset',
                             'baseline'],
                    default='all',
                    help='prior method: default=all')

parser.add_argument('--test_get_nears', action='store_true')
parser.add_argument('--baseline_cooccurrences', action='store_true')
parser.add_argument('--visualgenome', action='store_true')
parser.add_argument('--spatialsense', action='store_true')
parser.add_argument('--wordnet', action='store_true')
parser.add_argument('--dataset_stats', action='store_true')
parser.add_argument('--evaluate_fov_caches', action='store_true')
parser.add_argument('--dump_fov_caches', action='store_true')
parser.add_argument('--dump_oracle_caches', action='store_true')
parser.add_argument('--dump_fov_stats', action='store_true')

CAT2METHOD = {
    'all': ['wn',
            'word2vec',
            'glove',
            'fastText',
            'gpt',
            'gpt2',
            'gpt3',
            'gptneo',
            'roberta',
            'xlm',
            'ss',
            'vg'],
    'language':  ['wn',
                  'word2vec',
                  'glove',
                  'fastText',
                  'gpt',
                  'gpt2',
                  'gpt3',
                  'gptneo',
                  'roberta',
                  'xlm'],
    'vision': ['ss',
               'vg'],
    'lm':  ['gpt',
            'gpt2',
            'gpt3',
            'gptneo',
            'roberta',
            'xlm'],
    'word_emb': ['word2vec',
                 'glove',
                 'fastText']
}


def get_cooccurrence_files(method, cooccurrence_path, data_stats, version):
  cooccurrence_files = []

  if method in CAT2METHOD:
    for m in CAT2METHOD[method]:
      cooccurrence_files += [os.path.join(
          cooccurrence_path, 'cooccurrence.{}_{}.npy'.format(m, version))]
  if method == 'all' or method == 'dataset':
    data_stats_method = os.path.join(
        cooccurrence_path, 'cooccurrence.{}_{}.npy'.format(data_stats, version))
    cooccurrence_files += [data_stats_method]
  if method == 'baseline':
    baselines = ['uniform',
                 'random100',
                 'diagonal'
                 ]
    for baseline in baselines:
      cooccurrence_files += [os.path.join(
          cooccurrence_path, 'cooccurrence.{}.npy'.format(baseline))]
  return cooccurrence_files


args = parser.parse_args()

data_prefix = args.data_prefix
angle_inc = args.angle_inc
output_root = args.output_root
diag_mode = args.diag_mode
version = args.version
prefix_iou = args.prefix_iou
obj_dict_file = args.obj_dict_file
word_embedding_path = args.word_embedding_path
cooccurrence_path = args.cooccurrence_path
stats_path = args.stats_path

n_fovs = int((360 / angle_inc)*math.ceil(150/angle_inc))
butd_filename = './img_features/{}_{}degrees_obj36.tsv'.format(
    data_prefix,
    angle_inc)
cache_root = '{}_data/cached_data_{}degrees'.format(data_prefix, angle_inc)
image_list_file = './{}_data/imagelist.txt'.format(data_prefix)
iou = float('0.{}'.format(prefix_iou))
msuffix = 'mIOU{}'.format(prefix_iou)  # 'mHALF'

print('angle_inc', angle_inc)
print('n_fovs', n_fovs)
print('butd_filename', butd_filename)
print('cache_root', cache_root)
print('image_list_file', image_list_file)
print('output_root', output_root)
print('version:', version)
print('msuffix:', msuffix)
print('iou:', iou)
print('diag_mode:', diag_mode)
if args.test_get_nears:
  test_get_nears()
if args.baseline_cooccurrences:
  print('running generate_baseline_cooccurrences()')
  generate_baseline_cooccurrences(obj_dict_file=obj_dict_file,
                                  cooccurrence_path=cooccurrence_path)
if args.visualgenome:
  print('running get_visualgenome_stats()')
  get_visualgenome_stats(obj_dict_file=obj_dict_file,
                         cooccurrence_path=cooccurrence_path,
                         version=version)
if args.spatialsense:
  print('running get_spatialsense_stats()')
  get_spatialsense_stats(obj_dict_file=obj_dict_file,
                         cooccurrence_path=cooccurrence_path,
                         version=version)
if args.wordnet:
  print('running get_wordnet_stats()')
  get_wordnet_stats(obj_dict_file=obj_dict_file,
                    cooccurrence_path=cooccurrence_path,
                    version=version)
if args.dataset_stats:
  print('running get_dataset_stats()')
  get_dataset_stats(butd_filename=butd_filename,
                    image_list_file=image_list_file,
                    n_fovs=n_fovs,
                    angle_inc=angle_inc,
                    data_prefix=data_prefix,
                    obj_dict_file=obj_dict_file,
                    cooccurrence_path=cooccurrence_path,
                    version=version)

data_stats = '{}_d{}_butd'.format(data_prefix, angle_inc)
cooccurrence_files = get_cooccurrence_files(
    args.cooccurrence_method, cooccurrence_path, data_stats, version)
print('cooccurrence_files:')
pprint(cooccurrence_files)
if args.evaluate_fov_caches:
  print('running evaluate_fov_caches()')
  evaluate_fov_caches(
      cache_root=cache_root,
      butd_filename=butd_filename,
      image_list_file=image_list_file,
      n_fovs=n_fovs,
      angle_inc=angle_inc,
      data_prefix=data_prefix,
      word_embedding_path=word_embedding_path,
      obj_dict_file=obj_dict_file,
      cooccurrence_files=cooccurrence_files,
      output_root=output_root,
      diag_mode=diag_mode,
      msuffix=msuffix)

if args.dump_fov_caches:
  print('running dump_fov_caches()')
  dump_fov_caches(butd_filename=butd_filename,
                  image_list_file=image_list_file,
                  n_fovs=n_fovs,
                  angle_inc=angle_inc,
                  data_prefix=data_prefix,
                  word_embedding_path=word_embedding_path,
                  obj_dict_file=obj_dict_file,
                  cooccurrence_files=cooccurrence_files,
                  output_root=output_root,
                  diag_mode=diag_mode,
                  msuffix=msuffix)

if args.dump_oracle_caches:
  print('running dump_oracle_caches()')
  dump_oracle_caches(
      cache_root=cache_root,
      butd_filename=butd_filename,
      image_list_file=image_list_file,
      n_fovs=n_fovs,
      angle_inc=angle_inc,
      data_prefix=data_prefix,
      word_embedding_path=word_embedding_path,
      obj_dict_file=obj_dict_file,
      output_root=output_root)

if args.dump_fov_stats:
  print('running dump_fov_stats()')
  stats_files = [
      './stats/{}_cached{}degrees_stats.npy'.format(data_prefix, angle_inc)
  ]
  outfiles = [
      'img_features/{}_{}degrees_r{}statsall_v3.tsv'.format(
          data_prefix, angle_inc, angle_inc),
  ]
  methods = [
      'all_regions',
  ]
  dump_fov_stats(butd_filename=butd_filename,
                 image_list_file=image_list_file,
                 n_fovs=n_fovs,
                 angle_inc=angle_inc,
                 word_embedding_path=word_embedding_path,
                 obj_dict_file=obj_dict_file,
                 stats_files=stats_files,
                 outfiles=outfiles,
                 methods=methods)
