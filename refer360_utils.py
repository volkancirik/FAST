''' Utils for Refer360 grounding environment '''
import base64
from collections import defaultdict
import csv
import io
import json
from nltk.corpus import wordnet
import numpy as np
import os
from pprint import pprint
import sys
from tqdm import tqdm

from box_utils import get_boxes2coor_relationships
from box_utils import get_box2box_relationships
from box_utils import calculate_iou, calculate_area

file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(
    file_path, '..', '..', 'build_refer360'))
sys.path.append(module_path)
csv.field_size_limit(sys.maxsize)

EPS = 1e-15
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

MODEL2PREFIX = {'resnet': '',
                'clip': '.clip'}
MODEL2FEATURE_DIM = {'resnet': 2048,
                     'clip': 512}


def build_viewpoint_loc_embedding(viewIndex,
                                  angle_inc=15.0):
  '''
  Position embedding:
  heading 64D + elevation 64D
  1) heading: [sin(heading) for _ in range(1, 9)] +

              [cos(heading) for _ in range(1, 9)]
  2) elevation: [sin(elevation) for _ in range(1, 9)] +
                [cos(elevation) for _ in range(1, 9)]
  '''
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


def build_visited_embedding(adj_loc_list, visited):
  n_emb = 64
  half = int(n_emb/2)
  embedding = np.zeros((len(adj_loc_list), n_emb), np.float32)
  for kk, adj in enumerate(adj_loc_list):
    val = visited[adj['nextViewpointId']]
    embedding[kk,  0:half] = np.sin(val)
    embedding[kk, half:] = np.cos(val)
  return embedding


def build_oracle_embedding(adj_loc_list, gt_viewpoint_idx):
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


def load_cnn_features(image_list, cache_root, feature_prefix, n_fovs):

  print('cached features root:', cache_root)
  print('loading cnn features with prefix:', feature_prefix)
  pbar = tqdm(image_list)
  fov2feat = {}
  for fname in pbar:
    pano = fname.split('/')[-1].split('.')[0]
    feature_file = os.path.join(
        cache_root, 'features', '{}'.format(pano) + feature_prefix + '.npy')
    if not os.path.exists(feature_file):
      print('file missing:', feature_file)
      quit(0)

    fov_feats = np.load(feature_file).squeeze()

    for idx in range(n_fovs):
      pano_fov = '{}_{}'.format(pano, idx)
      fov2feat[pano_fov] = fov_feats[idx]
  return fov2feat


def load_butd(butd_filename,
              threshold=0.5,
              w2v=None,
              vg2name=None,
              keys=['features']):
  fov2key = {k: {} for k in keys}

  FIELDNAMES = ['img_id', 'img_h', 'img_w', 'objects_id', 'objects_conf',
                'attrs_id', 'attrs_conf', 'num_boxes', 'boxes', 'features']

  with open(butd_filename) as f:
    reader = csv.DictReader(f, FIELDNAMES, delimiter='\t')
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
        try:
          item[key] = item[key].reshape(shape)
        except:
          if key == 'boxes':
            dim = 4
          elif key == 'features':
            dim = 2048
          else:
            dim = 1
          item[key] = np.zeros((boxes, dim))
          pass
        item[key].setflags(write=False)

      keep_boxes = np.where(item['objects_conf'] >= threshold)[0]
      obj_ids = item['objects_id'][keep_boxes]

      scanId, viewpointId = item['img_id'].split('.')
      pano_fov = '{}_{}'.format(scanId, viewpointId)
      for k in keys:
        if k not in item:
          print('{} not in BUTD file'.format(k))
          quit(1)
        if k == 'features':
          feats = item['features'][keep_boxes]
          if w2v != None and vg2name != None:
            emb_feats = np.zeros((feats.shape[0], 300), dtype=np.float32)
            for ii, obj_id in enumerate(obj_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
            feats = emb_feats
          feats = np.sum(feats, axis=0)
          fov2key[k][pano_fov] = feats
        else:
          fov2key[k][pano_fov] = item[k][keep_boxes]
  return fov2key


def load_vectors(fname, vocab):
  fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  data = {}
  for line in fin:
    tokens = line.rstrip().split(' ')
    if tokens[0] not in vocab:
      continue
    data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
  return data


def get_object_dictionaries(obj_dict_file,
                            return_all=False):
  '''Loads object object dictionaries
  idx ->  visual genome
  visual genome -> idx
  idx -> object classes'''

  data = json.load(
      open(obj_dict_file, 'r'))
  vg2idx = data['vg2idx']
  idx2vg = data['idx2vg']
  obj_classes = data['obj_classes']

  if return_all:

    vg2idx = {int(k): int(vg2idx[k]) for k in vg2idx}
    idx2vg = {int(k): int(idx2vg[k]) for k in idx2vg}
    obj_classes.append('</s>')
    vg2idx[1601] = len(obj_classes)-1
    idx2vg[len(obj_classes)-1] = 1601

    name2vg, name2idx, vg2name = {}, {}, {}
    for idx in idx2vg:
      vg_idx = idx2vg[idx]
      obj_name = obj_classes[idx]

      name2vg[obj_name] = vg_idx
      name2idx[obj_name] = idx
      vg2name[vg_idx] = obj_name

    return vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name
  return vg2idx, idx2vg, obj_classes


def get_nears(boxes):
  n_boxes = len(boxes)
  inside = np.zeros((n_boxes, n_boxes))
  center = np.zeros((n_boxes, n_boxes))
  edge = np.ones((n_boxes, n_boxes))*400
  iou = np.zeros((n_boxes, n_boxes))
  for ii, b1 in enumerate(boxes):
    c = [(b1[1] + b1[3])/2, (b1[0] + b1[2])/2]
    r = get_boxes2coor_relationships(boxes[ii:], c)
    ious = []
    edges = []
    for jj, b2 in enumerate(boxes[ii:]):
      if ii == jj:
        area = 0
      else:
        area = calculate_iou(b1, b2)
      ious += [area]
      e, _ = get_box2box_relationships(b1, b2)
      edges += [e]
    inside[ii, ii:] = r[0]
    center[ii, ii:] = r[1]
    iou[ii, ii:] = ious
    edge[ii, ii:] = edges

  nears = []
  for ii in range(n_boxes):
    for jj in range(ii+1, n_boxes):
      near = False
      if iou[ii, jj] > 0.0 or center[ii, jj] < 100:
        near = True
      else:
        if edge[ii, jj] < 20:
          near = True
        if center[ii, jj] < 40:
          near = True
        # add other conditions to update near
      if near:
        nears.append([ii, jj])
  return nears, inside, center, iou, edge


def test_get_nears():
  obj_dict_file = './tasks/FAST/data/vg_object_dictionaries.all.json'
  boxes = np.array([[87.36437, 17.067612, 272.22318, 172.34209],
                    [0.,       288.84726,   94.519585, 395.91797],
                    [0.,        35.797077, 317.38986,  399.33334],
                    [50, 50, 300, 300],
                    [309.07068,  118.77045,  394.10828,  216.93341],
                    [175.26164,  361.52023,  244.46167,  398.51883]])
  object_ids = [743, 177, 72, 99, 781, 781]

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  print('# of objects:', len(vg2name), len(obj_classes))
  nears, inside, centers, iou, edge = get_nears(boxes)
  print('nears\n', nears)
  print('_'*20)
  print('inside\n', inside)
  print('_'*20)
  print('centers\n', centers)
  print('_'*20)
  print('iou\n', iou)
  print('_'*20)
  print('edge\n', edge)
  print('_'*20)
  n_objects = len(vg2name)
  cooccurrence = np.zeros((n_objects, n_objects))
  for near in nears:
    o1, o2 = object_ids[near[0]], object_ids[near[1]]
    name1, name2 = vg2name.get(o1, '</s>'), vg2name.get(o2, '</s>')
    idx1, idx2 = obj_classes.index(name1), obj_classes.index(name2)
    idx3, idx4 = vg2idx.get(
        o1, name2idx['</s>']), vg2idx.get(o2, name2idx['</s>'])
    print('nears:', near[0], near[1], o1, o2,
          idx1, idx3, idx2, idx4, name1, name2)
    cooccurrence[idx1, idx2] += 1
    cooccurrence[idx2, idx1] += 1
  print(cooccurrence[idx1, :])
  d = {'method': 'test',
       'cooccurrence': cooccurrence}
  np.save('./cooccurrences/cooccurrence.test.npy', d)
  d = np.load('./cooccurrences/cooccurrence.test.npy', allow_pickle=True)[()]
  cooccurrence = d['cooccurrence']
  method = d['method']
  print(method)
  print(cooccurrence[idx1, :])


def get_dataset_stats(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        data_prefix='refer360',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_path='./cooccurrences',
        version='v3'):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])

  n_objects = len(vg2name)
  cooccurrence = np.zeros((n_objects, n_objects))
  distance_mean = np.zeros((n_objects, n_objects))
  distance_sigma = np.zeros((n_objects, n_objects))
  distance_list = {obj: defaultdict(list) for obj in obj_classes}
  print('loaded BUTD boxes!', image_list_file)
  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  image_list = [line.strip()
                for line in open(image_list_file)]
  pbar = tqdm(image_list)

  for fname in pbar:
    pano = fname.split('/')[-1].split('.')[0]
    for idx in range(n_fovs):
      pano_fov = '{}_{}'.format(pano, idx)
      if pano_fov not in fov2keys['boxes'] or pano_fov not in fov2keys['objects_id']:
        continue
      boxes = fov2keys['boxes'][pano_fov]
      object_ids = fov2keys['objects_id'][pano_fov]
      nears, inside, centers, iou, edge = get_nears(boxes)
      for near in nears:
        o1, o2 = object_ids[near[0]], object_ids[near[1]]
        name1, name2 = vg2name.get(o1, '</s>'), vg2name.get(o2, '</s>')
        idx1, idx2 = obj_classes.index(name1), obj_classes.index(name2)
        cooccurrence[idx1, idx2] += 1
        cooccurrence[idx2, idx1] += 1

      for kk in range(len(centers)):
        for jj in range(kk+1, len(centers)):
          o1, o2 = object_ids[kk], object_ids[jj]
          name1, name2 = vg2name.get(o1, '</s>'), vg2name.get(o2, '</s>')
          idx1, idx2 = obj_classes.index(name1), obj_classes.index(name2)
          distance = centers[kk][jj]
          distance_list[name1][name2].append(distance)
          distance_list[name2][name1].append(distance)

  for kk, src in enumerate(obj_classes):
    for jj, trg in enumerate(obj_classes):
      idx1, idx2 = obj_classes.index(src), obj_classes.index(trg)
      mean, sigma = 0, 0
      if len(distance_list[src][trg]) > 1:
        mean = np.mean(distance_list[src][trg])
        sigma = np.std(distance_list[src][trg])
      distance_mean[idx1][idx2] = mean
      distance_sigma[idx1][idx2] = sigma

  d = {'method': '{}_{}degrees_butd_36obj'.format(data_prefix, angle_inc),
       'prefix': 'r{}butd_{}'.format(angle_inc, version),
       'butd_filename': butd_filename,
       'cooccurrence': cooccurrence,
       'distance_mean': distance_mean,
       'distance_sigma': distance_sigma}

  out_file = os.path.join(cooccurrence_path,
                          'cooccurrence.{}_d{}_butd_{}.npy'.format(data_prefix,
                                                                   angle_inc,
                                                                   version))
  print('dumping to', out_file)
  np.save(out_file, d)
  print('DONE! bye.')


def get_spatialsense_stats(
        spatialsense_annotations='/projects3/all_data/spatialsense/annotations.json',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_path='./cooccurrences',
        version='v3'):

  anns = json.load(open(spatialsense_annotations, 'r'))

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  cooccurrence = np.zeros((n_objects, n_objects))
  distance_mean = np.zeros((n_objects, n_objects))
  distance_sigma = np.zeros((n_objects, n_objects))
  distance_list = {obj: defaultdict(list) for obj in obj_classes}

  pbar = tqdm(anns)
  for imgs in pbar:

    for rel in imgs['annotations']:
      name1 = rel['object']['name'].lower()
      name2 = rel['subject']['name'].lower()
      box1 = [rel['object']['bbox'][0], rel['object']['bbox'][1], rel['object']['bbox']
              [0]+rel['object']['bbox'][2], rel['object']['bbox'][1]+rel['object']['bbox'][3]]
      box2 = [rel['subject']['bbox'][0], rel['subject']['bbox'][1], rel['subject']['bbox']
              [0]+rel['subject']['bbox'][2], rel['subject']['bbox'][1]+rel['subject']['bbox'][3]]
      area1 = calculate_area(box1)
      area2 = calculate_area(box2)
      if name1 not in name2vg or area1 < 20:  # arbitrary number
        continue
      if name2 not in name2vg or area2 < 20:  # arbitrary number
        continue

      idx1 = obj_classes.index(name1)
      idx2 = obj_classes.index(name2)
      object_ids, boxes = [idx1, idx2], [box1, box2]
      nears, inside, centers, iou, _ = get_nears(boxes)
      for near in nears:
        idx1, idx2 = object_ids[near[0]], object_ids[near[1]]
        cooccurrence[idx1, idx2] += 1
        cooccurrence[idx2, idx1] += 1

      for kk in range(len(centers)):
        for jj in range(kk+1, len(centers)):
          o1, o2 = object_ids[kk], object_ids[jj]
          name1, name2 = vg2name.get(o1, '</s>'), vg2name.get(o2, '</s>')
          idx1, idx2 = obj_classes.index(name1), obj_classes.index(name2)
          distance = centers[kk][jj]
          distance_list[name1][name2].append(distance)
          distance_list[name2][name1].append(distance)

  for kk, src in enumerate(obj_classes):
    for jj, trg in enumerate(obj_classes):
      idx1, idx2 = obj_classes.index(src), obj_classes.index(trg)
      mean, sigma = 0, 0
      if len(distance_list[src][trg]) > 1:
        mean = np.mean(distance_list[src][trg])
        sigma = np.std(distance_list[src][trg])
      distance_mean[idx1][idx2] = mean
      distance_sigma[idx1][idx2] = sigma

  d = {'method': 'spatialsense',
       'prefix': 'ss_{}'.format(version),
       'cooccurrence': cooccurrence,
       'distance_mean': distance_mean,
       'distance_sigma': distance_sigma}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.ss_{}.npy'.format(version)), d)
  print('DONE! bye.')


def get_visualgenome_stats(
        visualgenome_objects='/projects3/all_data/visualgenome/objects.json',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_path='./cooccurrences',
        version='v3'):

  objects = json.load(open(visualgenome_objects, 'r'))

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  cooccurrence = np.zeros((n_objects, n_objects))
  distance_mean = np.zeros((n_objects, n_objects))
  distance_sigma = np.zeros((n_objects, n_objects))
  distance_list = {obj: defaultdict(list) for obj in obj_classes}

  pbar = tqdm(objects)
  for imgs in pbar:
    object_ids, boxes = [], []
    for obj in imgs['objects']:
      name = obj['names'][0]
      box = [obj['x'], obj['y'], obj['x']+obj['w'], obj['y']+obj['h']]
      area = calculate_area(box)
      if name not in name2vg or area < 20:  # arbitrary number
        continue
      idx = obj_classes.index(name)
      object_ids.append(idx)
      boxes.append(box)
    nears, inside, centers, iou, _ = get_nears(boxes)
    for near in nears:
      idx1, idx2 = object_ids[near[0]], object_ids[near[1]]
      cooccurrence[idx1, idx2] += 1
      cooccurrence[idx2, idx1] += 1
    for kk in range(len(centers)):
      for jj in range(kk+1, len(centers)):
        o1, o2 = object_ids[kk], object_ids[jj]
        name1, name2 = vg2name.get(o1, '</s>'), vg2name.get(o2, '</s>')
        idx1, idx2 = obj_classes.index(name1), obj_classes.index(name2)
        distance = centers[kk][jj]
        distance_list[name1][name2].append(distance)
        distance_list[name2][name1].append(distance)

  for kk, src in enumerate(obj_classes):
    for jj, trg in enumerate(obj_classes):
      idx1, idx2 = obj_classes.index(src), obj_classes.index(trg)
      mean, sigma = 0, 0
      if len(distance_list[src][trg]) > 1:
        mean = np.mean(distance_list[src][trg])
        sigma = np.std(distance_list[src][trg])
      distance_mean[idx1][idx2] = mean
      distance_sigma[idx1][idx2] = sigma

  d = {'method': 'visualgenome v1.4',
       'prefix': 'vg_{}'.format(version),
       'cooccurrence': cooccurrence,
       'distance_mean': distance_mean,
       'distance_sigma': distance_sigma}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.vg_{}.npy'.format(version)), d)
  print('DONE! bye.')


def wordnet_similarity(word1, word2):
  synsets1 = wordnet.synsets(word1)
  synsets2 = wordnet.synsets(word2)
  if synsets1 == [] or synsets2 == []:
    return 0
  wordFromList1 = synsets1[0]
  wordFromList2 = synsets2[0]
  return wordFromList1.wup_similarity(wordFromList2)


def get_wordnet_stats(
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_path='./cooccurrences',
        version='v3'):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  cooccurrence = np.zeros((n_objects, n_objects))
  for name1 in name2vg.keys():
    for name2 in name2vg.keys():
      sim_score = wordnet_similarity(name1, name2)
      idx1 = obj_classes.index(name1)
      idx2 = obj_classes.index(name2)
      cooccurrence[idx1, idx2] = sim_score
      cooccurrence[idx2, idx1] = sim_score

  d = {'method': 'wordnet similarity',
       'prefix': 'wn_{}'.format(version),
       'cooccurrence': cooccurrence}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.wn_{}.npy'.format(version)), d)
  print('DONE! bye.')


def get_points(radius, number_of_points):
  radians_between_each_point = 2*np.pi/number_of_points
  list_of_points = []
  for p in range(0, number_of_points):
    list_of_points.append((radius*np.cos(p*radians_between_each_point),
                           radius*np.sin(p*radians_between_each_point)))
  return list_of_points


def gauss_map_fast(gt_x, gt_y,
                   mean=50,
                   sigma=3.0,
                   width=400,
                   height=400,
                   mean_x=None,
                   mean_y=None,
                   sigma_x=None,
                   sigma_y=None):
  if sigma_x == None:
    sigma_x = sigma
  if sigma_y == None:
    sigma_y = sigma
  if mean_x == None:
    mean_x = mean
  if mean_y == None:
    mean_y = mean

  assert isinstance(width, int)
  assert isinstance(height, int)

  x0 = width // 2
  y0 = height // 2

  x = np.arange(0, width, dtype=float)
  y = np.arange(0, height, dtype=float)[:, np.newaxis]

  x -= x0
  y -= y0
  d = np.sqrt(x*x+y*y)

  exp_part = (d-mean_x)**2/(2*sigma_x**2) + (d-mean_y)**2/(2*sigma_y**2)
  gaussian = 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-exp_part)
  full = np.zeros((width*3, height*3))
  full[height:height*2, width:width*2] = gaussian

  start_y = height + int((height/2 - gt_y))
  start_x = width + int((width/2 - gt_x))
  crop = full[start_y:start_y+height, start_x:start_x+width]
  return crop


def get_ring_fast(width=400, height=400, mean=100, sigma=10, center_x=-1, center_y=-1):
  if center_x < 0:
    center_x = width/2
  if center_y < 0:
    center_y = height/2
  canvas = gauss_map_fast(center_x, center_y,
                          width=width, height=height,
                          sigma=sigma,
                          mean=mean)
  return canvas


def get_canvas(object_tuples, distance_mean, distance_sigma,
               canvas_ratio=20,
               fov_size=400):
  diff = int(fov_size/canvas_ratio)
  canvas_size = diff*3
  canvas = np.zeros((canvas_size, canvas_size, distance_mean.shape[1]))
  # canvas = np.zeros((canvas_size,canvas_size))

  for ii, (idx, o) in enumerate(object_tuples):
    x, y, w, h = int(o[0]/canvas_ratio)+diff, int(o[1]/canvas_ratio) + \
        diff, int(
        (o[2]-o[0])/canvas_ratio), int((o[3]-o[1])/canvas_ratio)
    center_x = x + w/2
    center_y = y + h/2
    for jj in range(distance_mean.shape[1]):
      d_mean = distance_mean[idx][jj] / canvas_ratio
      d_sigma = distance_sigma[idx][jj] / canvas_ratio
      if d_mean == 0 or d_sigma == 0:
        continue
      ring = get_ring_fast(width=canvas_size,
                           height=canvas_size,
                           mean=d_mean,
                           sigma=d_sigma,
                           center_x=center_x,
                           center_y=center_y)
      # canvas += ring
      canvas[:, :, jj] += ring
  return canvas


DIR2RANGES = {
    'ul': (0, 0, 1, 1),
    'u': (1, 0, 2, 1),
    'ur': (2, 0, 3, 1),
    'l': (0, 1, 1, 2),
    'r': (2, 1, 3, 2),
    'dl': (0, 2, 1, 3),
    'd': (1, 2, 2, 3),
    'dr': (2, 2, 3, 3),
}


def get_direction_weights(canvas):
  coeff = canvas.shape[0]/3

  weights = {direction: np.array([0.0]*canvas.shape[2])
             for direction in DIR2RANGES}
  for direction in DIR2RANGES:
    min_x = int(DIR2RANGES[direction][0]*coeff)
    min_y = int(DIR2RANGES[direction][1]*coeff)
    max_x = int(DIR2RANGES[direction][2]*coeff)
    max_y = int(DIR2RANGES[direction][3]*coeff)
    for obj in range(canvas.shape[2]):
      weights[direction][obj] = np.sum(canvas[min_y:max_y, min_x:max_x, obj])
    sum_count = np.sum(weights[direction])
    if sum_count > 0:
      weights[direction] = weights[direction] / sum_count
  return weights


def dump_gaussian_caches(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        data_prefix='refer360',
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
    output_root='./img_features',
        cooccurrence_files=[],
        msuffix=''):
  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)
  FIELDNAMES = ['pano_fov', 'features']

  for cooccurrence_file in cooccurrence_files:

    cooccurrence_data = np.load(cooccurrence_file,
                                allow_pickle=True)[()]
    distance_mean = cooccurrence_data['distance_mean']
    distance_sigma = cooccurrence_data['distance_sigma']
    prefix = cooccurrence_data['prefix']

    suffix = msuffix
    outfile = os.path.join(output_root, '{}_{}degrees_{}{}.tsv'.format(
        data_prefix, angle_inc, prefix, suffix))
    print('output file:', outfile)

    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    with open(outfile, 'w') as tsvfile:
      writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
      for fname in pbar:
        pano = fname.split('/')[-1].split('.')[0]
        for idx in range(n_fovs):
          pano_fov = '{}_{}'.format(pano, idx)
          features = np.zeros((9, 300), dtype=np.float32)

          if pano_fov in fov2keys['boxes'] and pano_fov in fov2keys['objects_id']:
            boxes = fov2keys['boxes'][pano_fov]
            object_ids = fov2keys['objects_id'][pano_fov]
            n_boxes = len(boxes)

            emb_feats = np.zeros((n_boxes, 300), dtype=np.float32)
            object_tuples = []
            for ii, obj_id in enumerate(object_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
              idx = obj_classes.index(obj_name)
              object_tuples.append((idx, boxes[ii]))

            canvas = get_canvas(object_tuples, distance_mean, distance_sigma)
            weights = get_direction_weights(canvas)
            features[4, :] = np.sum(emb_feats, axis=0)

            # for each direction in ul, u, ur, l, r, dl, d, dr
            for direction in DIR2IDX:
              dir_feats = np.zeros((1, 300), dtype=np.float32)
              feat_index = DIR2IDX[direction]

              # for each object on the edge
              for obj in obj_classes:
                idx = obj_classes.index(obj)
                if weights[direction][idx] > 0:
                  emb = w2v.get(obj, w2v['</s>'])
                  dir_feats += emb * weights[direction][idx]
              features[feat_index, :] = dir_feats
          encoded = base64.b64encode(features).decode()
          d = {'pano_fov': pano_fov,
               'features': encoded}

          writer.writerow(d)
    pbar.close()
    print('DONE!')
  print('DONE with all!')


def dump_fov_caches(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        data_prefix='refer360',
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_files=[],
        output_root='./img_features',
        diag_mode=-1,
        msuffix=''):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)
  FIELDNAMES = ['pano_fov', 'features']

  for cooccurrence_file in cooccurrence_files:

    cooccurrence_data = np.load(cooccurrence_file,
                                allow_pickle=True)[()]
    cooccurrence = cooccurrence_data['cooccurrence']

    cooccurrence_data = np.load(cooccurrence_file,
                                allow_pickle=True)[()]
    cooccurrence = cooccurrence_data['cooccurrence']
    # normalize the counts
    normalize_column = 'prompt' in cooccurrence_data['method']
    if normalize_column:
      print('will normalize columns')
      for idx in range(cooccurrence.shape[0]):
        sum_count = np.sum(cooccurrence[:, idx])
        if sum_count > 0:
          cooccurrence[:, idx] = cooccurrence[:, idx] / sum_count

    for idx in range(cooccurrence.shape[0]):
      sum_count = np.sum(cooccurrence[idx, :])
      if sum_count > 0:
        cooccurrence[idx, :] = cooccurrence[idx, :] / sum_count

    suffix = msuffix
    if diag_mode == 0:
      suffix += 'DIAG0'
      np.fill_diagonal(cooccurrence, 0)
    elif diag_mode == 1:
      suffix += 'DIAG1'
      np.fill_diagonal(cooccurrence, 1)
    elif diag_mode == -1:
      pass
    else:
      raise NotImplementedError()

    prefix = cooccurrence_data['prefix']
    outfile = os.path.join(output_root, '{}_{}degrees_{}{}.tsv'.format(
        data_prefix, angle_inc, prefix, suffix))
    print('output file:', outfile)

    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    with open(outfile, 'w') as tsvfile:
      writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
      for fname in pbar:
        pano = fname.split('/')[-1].split('.')[0]
        for idx in range(n_fovs):
          pano_fov = '{}_{}'.format(pano, idx)
          features = np.zeros((9, 300), dtype=np.float32)

          if pano_fov in fov2keys['boxes'] and pano_fov in fov2keys['objects_id']:
            boxes = fov2keys['boxes'][pano_fov]
            object_ids = fov2keys['objects_id'][pano_fov]
            n_boxes = len(boxes)

            emb_feats = np.zeros((n_boxes, 300), dtype=np.float32)
            for ii, obj_id in enumerate(object_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
            features[4, :] = np.sum(emb_feats, axis=0)
            dir2obj = defaultdict(list)

            # TODO: this should be a function
            for ii, box in enumerate(boxes):
              directions = []
              # box[1] < 120:  # 200:
              if calculate_iou(box, [0, 0, 400, 120]) > 0.1:
                directions.append('u')
              # box[1] > 280:  # 200:
              elif calculate_iou(box, [0, 280, 400, 400]) > 0.1:
                directions.append('d')
              # box[0] < 120:  # 200:
              if calculate_iou(box, [0, 0, 120, 400]) > 0.1:
                if directions:
                  directions[0] += 'l'
                directions.append('l')
              # box[0] > 280:  # 200:
              elif calculate_iou(box, [280, 0, 400, 400]) > 0.1:
                if directions:
                  directions[0] += 'r'
                directions.append('r')
              for direction in directions:
                dir2obj[direction] += [ii]

            # for each direction in ul, u, ur, l, r, dl, d, dr
            for direction in dir2obj.keys():
              indexes = dir2obj[direction]
              feat_index = DIR2IDX[direction]
              dir_feats = np.zeros((1, 300), dtype=np.float32)

              # for each object on the edge
              for index in indexes:
                o = object_ids[index]
                name = vg2name.get(o, '</s>')
                idx = obj_classes.index(name)

                for co_idx in range(cooccurrence.shape[1]):
                  if cooccurrence[idx, co_idx] > 0:
                    co_name = obj_classes[co_idx]
                    emb = w2v.get(co_name, w2v['</s>'])
                    dir_feats += emb * cooccurrence[idx, co_idx]
              features[feat_index, :] = dir_feats
          encoded = base64.b64encode(features).decode()
          # TODO: d should include dir2obj
          d = {'pano_fov': pano_fov,
               'features': encoded}

          writer.writerow(d)
    pbar.close()
    print('DONE!')
  print('DONE with all!')


def dump_oracle_caches(
        cache_root='refer360_data/cached_data_30degrees',
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        data_prefix='refer360',
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        output_root='./img_features'):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)

  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']

  FIELDNAMES = ['pano_fov', 'features']

  recall_prec = [(1.00, v) for v in [0.25, 0.50, 0.75, 1.00][::-1]]
  recall_prec += [(v, 1.00) for v in [0.25, 0.50, 0.75][::-1]]

  for oracle_rec_rate, oracle_prec_rate in recall_prec:

    obj_tp = defaultdict(float)
    obj_fp = defaultdict(float)
    obj_fn = defaultdict(float)
    obj_tn = defaultdict(float)

    all_loss = 0.0
    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0
    all_tn = 0.0

    outfile = os.path.join(output_root, '{}_{}degrees_oracle_prec{}_rec{}.tsv'.format(
        data_prefix, angle_inc, oracle_prec_rate, oracle_rec_rate))
    print('output file:', outfile)

    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    with open(outfile, 'w') as tsvfile:
      writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
      for fname in pbar:
        pano = fname.split('/')[-1].split('.')[0]
        for idx in range(n_fovs):
          pano_fov = '{}_{}'.format(pano, idx)
          features = np.zeros((9, 300), dtype=np.float32)

          if pano_fov in fov2keys['boxes'] and pano_fov in fov2keys['objects_id']:
            boxes = fov2keys['boxes'][pano_fov]
            object_ids = fov2keys['objects_id'][pano_fov]
            n_boxes = len(boxes)

            emb_feats = np.zeros((n_boxes, 300), dtype=np.float32)
            for ii, obj_id in enumerate(object_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
            features[4, :] = np.sum(emb_feats, axis=0)

            for neighbor in nodes[idx]['neighbor2dir']:
              direction = nodes[idx]['neighbor2dir'][neighbor]
              gt_boxes = [vg2idx.get(vg, obj_classes.index(
                  '</s>')) for vg in fov2keys['objects_id']['{}_{}'.format(pano, neighbor)]]

              pred_boxes = np.zeros(
                  (1, n_objects), dtype=np.float32)
              dir_feats = np.zeros((1, 300), dtype=np.float32)

              pos_count = 0
              neg_count = 0
              for src_idx in range(n_objects):
                r = np.random.uniform()
                truth = int(src_idx in gt_boxes)

                if truth:
                  if r < oracle_rec_rate:
                    pred_boxes[0, src_idx] = truth
                    pos_count += 1
                  else:
                    pred_boxes[0, src_idx] = 1-truth
                else:
                  if r < 1-oracle_prec_rate and ((1-oracle_prec_rate)*pos_count)/oracle_prec_rate >= neg_count:
                    pred_boxes[0, src_idx] = 1-truth
                    neg_count += 1

                if pred_boxes[0, src_idx] > 0:
                  co_name = obj_classes[src_idx]
                  emb = w2v.get(co_name, w2v['</s>'])
                  dir_feats += emb * pred_boxes[0, src_idx]

              feat_index = DIR2IDX[direction]
              features[feat_index, :] = dir_feats

              # calculate metrics
              for obj_idx in range(n_objects-1):
                predicted = float(pred_boxes[0, obj_idx] > 0.5)
                if obj_idx in gt_boxes:
                  all_loss += logloss(1, pred_boxes[0, obj_idx])
                  if predicted:
                    obj_tp[obj_idx] += 1.0
                    all_tp += 1.0
                  else:
                    obj_fn[obj_idx] += 1.0
                    all_fn += 1.0
                else:
                  all_loss += logloss(0, pred_boxes[0, obj_idx])
                  if predicted:
                    obj_fp[obj_idx] += 1.0
                    all_fp += 1.0
                  else:
                    obj_tn[obj_idx] += 1.0
                    all_tn += 1.0
          encoded = base64.b64encode(features).decode()
          d = {'pano_fov': pano_fov,
               'features': encoded}
          writer.writerow(d)
    pbar.close()
    if (all_tp + all_fp) == 0:
      all_precision = 0
    else:
      all_precision = all_tp / (all_tp + all_fp)
    if (all_tp + all_fn) == 0:
      all_recall = 0
    else:
      all_recall = all_tp / (all_tp + all_fn)
    if (all_precision + all_recall) == 0:
      all_f1 = 0
    else:
      all_f1 = 2 * (all_precision * all_recall) / \
          (all_precision + all_recall)
    print('Oracle prec rate {} oracle rec rate {}\nprec {} recall {} f1 {} loss {}'.format(
        oracle_prec_rate,
        oracle_rec_rate,
        all_precision,
        all_recall,
        all_f1,
        all_f1))
    print('tp fp fn tn', all_tp, all_fp, all_fn, all_tn)

    print('DONE!')
  print('DONE with all!')


def logloss(true_label, predicted, eps=EPS):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)


def evaluate_fov_caches(
        cache_root='refer360_data/cached_data_30degrees',
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        data_prefix='refer360',
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_files=[],
        output_root='./img_features',
        diag_mode=-1,
        msuffix=''):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  # n_objects = len(vg2name)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)

  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']

  for cooccurrence_file in cooccurrence_files:

    cooccurrence_data = np.load(cooccurrence_file,
                                allow_pickle=True)[()]
    cooccurrence = cooccurrence_data['cooccurrence']

    # normalize the counts
    normalize_column = 'prompt' in cooccurrence_data['method']
    if normalize_column:
      print('will normalize columns')
      for idx in range(cooccurrence.shape[0]):
        sum_count = np.sum(cooccurrence[:, idx])
        if sum_count > 0:
          cooccurrence[:, idx] = cooccurrence[:, idx] / sum_count

    for idx in range(cooccurrence.shape[0]):
      sum_count = np.sum(cooccurrence[idx, :])
      if sum_count > 0:
        cooccurrence[idx, :] = cooccurrence[idx, :] / sum_count

    suffix = msuffix
    if diag_mode == 0:
      suffix += 'DIAG0'
      np.fill_diagonal(cooccurrence, 0)
    elif diag_mode == 1:
      suffix += 'DIAG1'
      np.fill_diagonal(cooccurrence, 1)
    elif diag_mode == -1:
      pass
    else:
      raise NotImplementedError()

    prefix = cooccurrence_data['prefix']
    evalfile = os.path.join(output_root, '{}_{}degrees_{}{}.eval.npy'.format(
        data_prefix, angle_inc, prefix, suffix))
    print('stats for {}: {:2.3f} {:2.3f} {:2.3f} {:2.3f} {:2.3f}'.format(prefix,
                                                                         np.sum(
                                                                             cooccurrence[:, :]),
                                                                         np.mean(
                                                                             cooccurrence[:, :]),
                                                                         np.median(
                                                                             cooccurrence[:, :]),
                                                                         np.min(
                                                                             cooccurrence[:, :]),
                                                                         np.max(cooccurrence[:, :])))
    print('eval output file:', evalfile)

    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    obj_tp = defaultdict(float)
    obj_fp = defaultdict(float)
    obj_fn = defaultdict(float)
    obj_tn = defaultdict(float)

    all_loss = 0.0
    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0
    all_tn = 0.0

    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]
      for idx in range(n_fovs):
        pano_fov = '{}_{}'.format(pano, idx)

        if pano_fov in fov2keys['boxes'] and pano_fov in fov2keys['objects_id']:
          boxes = fov2keys['boxes'][pano_fov]
          object_ids = fov2keys['objects_id'][pano_fov]

          dir2obj = defaultdict(list)
          for ii, box in enumerate(boxes):
            o = object_ids[ii]
            name = vg2name.get(o, '</s>')
            src_idx = obj_classes.index(name)

            directions = []
            if calculate_iou(box, [0, 0, 400, 120]) > 0.1:  # box[1] < 200:
              directions.append('u')
            elif calculate_iou(box, [0, 280, 400, 400]) > 0.1:  # box[1] > 200:
              directions.append('d')
            if calculate_iou(box, [0, 0, 120, 400]) > 0.1:  # box[0] < 200:
              if directions:
                directions[0] += 'l'
              directions.append('l')
            elif calculate_iou(box, [280, 0, 400, 400]) > 0.1:  # box[0] > 200:
              if directions:
                directions[0] += 'r'
              directions.append('r')
            for direction in directions:
              dir2obj[direction] += [src_idx]

          # for each direction in ul, u, ur, l, r, dl, d, dr
          for neighbor in nodes[idx]['neighbor2dir']:
            direction = nodes[idx]['neighbor2dir'][neighbor]
            gt_boxes = [vg2idx.get(vg, obj_classes.index(
                '</s>')) for vg in fov2keys['objects_id']['{}_{}'.format(pano, neighbor)]]

            indexes = dir2obj[direction]

            pred_boxes = np.zeros(
                (1, cooccurrence.shape[1]), dtype=np.float32)
            for src_idx in indexes:
              for co_idx in range(cooccurrence.shape[1]):
                pred_boxes[0, co_idx] += cooccurrence[src_idx, co_idx]

            for obj_idx in range(pred_boxes.shape[1]-1):
              predicted = float(pred_boxes[0, obj_idx] > 0.0)
              if obj_idx in gt_boxes:
                all_loss += logloss(1, pred_boxes[0, obj_idx])
                if predicted:
                  obj_tp[obj_idx] += 1.0
                  all_tp += 1.0
                else:
                  obj_fn[obj_idx] += 1.0
                  all_fn += 1.0
              else:
                # all_loss += logloss(0, pred_boxes[0, obj_idx])
                if predicted:
                  obj_fp[obj_idx] += 1.0
                  all_fp += 1.0
                else:
                  obj_tn[obj_idx] += 1.0
                  all_tn += 1.0
    obj_recall = {}
    obj_precision = {}
    obj_f1 = {}
    for obj in obj_tp:
      if (obj_tp[obj]+obj_fp[obj]) == 0:
        obj_precision[obj] = 0
      else:
        obj_precision[obj] = obj_tp[obj] / (obj_tp[obj]+obj_fp[obj])
      if (obj_tp[obj]+obj_fn[obj]) == 0:
        obj_recall[obj] = 0
      else:
        obj_recall[obj] = obj_tp[obj] / (obj_tp[obj]+obj_fn[obj])
      if (obj_precision[obj] + obj_recall[obj]) == 0:
        obj_f1[obj] = 0
      else:
        obj_f1[obj] = 2 * (obj_precision[obj] * obj_recall[obj]) / \
            (obj_precision[obj] + obj_recall[obj])
    if (all_tp + all_fp) == 0:
      all_precision = 0
    else:
      all_precision = all_tp / (all_tp + all_fp)
    if (all_tp + all_fn) == 0:
      all_recall = 0
    else:
      all_recall = all_tp / (all_tp + all_fn)
    if (all_precision + all_recall) == 0:
      all_f1 = 0
    else:
      all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall)
    eval_stats = {
        'obj_precision': obj_precision,
        'obj_recall': obj_recall,
        'obj_f1': obj_f1,
        'all_precision': all_precision,
        'all_recall': all_recall,
        'all_f1': all_f1,
        'all_loss': all_loss
    }

    np.save(evalfile, eval_stats)
    pbar.close()
    print('prec {:2.3f} recall {:2.3f} f1 {:2.3f} loss {:2.3f}'.format(all_precision,
                                                                       all_recall,
                                                                       all_f1,
                                                                       all_f1))
  print('DONE with all!')


def dump_fov_stats(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        angle_inc=30,
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        stats_files=[],
        methods=[],
        outfiles=[]):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)

  FIELDNAMES = ['pano_fov', 'features']

  assert len(stats_files) == len(outfiles)
  assert len(methods) == len(outfiles)
  for stats_file, outfile, method in zip(stats_files, outfiles, methods):
    stats_data = np.load(stats_file,
                         allow_pickle=True)[()][method]

    print('output file:', outfile)

    image_list = [line.strip()
                  for line in open(image_list_file)]
    pbar = tqdm(image_list)

    with open(outfile, 'w') as tsvfile:
      writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
      for fname in pbar:
        pano = fname.split('/')[-1].split('.')[0]
        for idx in range(n_fovs):
          pano_fov = '{}_{}'.format(pano, idx)
          features = np.zeros((9, 300), dtype=np.float32)

          if pano_fov in fov2keys['boxes'] and pano_fov in fov2keys['objects_id']:
            boxes = fov2keys['boxes'][pano_fov]
            object_ids = fov2keys['objects_id'][pano_fov]
            n_boxes = len(boxes)

            emb_feats = np.zeros((n_boxes, 300), dtype=np.float32)
            for ii, obj_id in enumerate(object_ids):
              obj_name = vg2name.get(obj_id, '</s>')
              emb_feats[ii, :] = w2v.get(obj_name, w2v['</s>'])
            features[4, :] = np.sum(emb_feats, axis=0)

            if method == 'all_regions':
              stats = stats_data['navigation']
            else:
              stats = stats_data[idx]['navigation']

            for direction in stats.keys():
              row = stats[direction]
              feat_index = DIR2IDX[direction]
              dir_feats = np.zeros((1, 300), dtype=np.float32)

              for co_idx in range(n_objects):
                if row[co_idx] > 0:
                  co_name = obj_classes[co_idx]
                  emb = w2v.get(co_name, w2v['</s>'])
                  dir_feats += emb * row[co_idx]
              features[feat_index, :] = dir_feats
          encoded = base64.b64encode(features).decode()
          d = {'pano_fov': pano_fov,
               'features': encoded}
          writer.writerow(d)
    pbar.close()
    print('DONE!')
  print('DONE with all!')


def generate_baseline_cooccurrences(
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_path='./cooccurrences'):

  _, _, _, _, _, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)
  rand100_cooccurrence = np.random.randint(100, size=(n_objects, n_objects))

  d = {'method': 'ranindt max 100',
       'prefix': 'random100',
       'cooccurrence': rand100_cooccurrence}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.random100.npy'), d)

  uniform_cooccurrence = np.ones((n_objects, n_objects))
  d = {'method': 'uniform',
       'prefix': 'uniform',
       'cooccurrence': uniform_cooccurrence}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.uniform.npy'), d)

  diagonal_cooccurrence = np.zeros((n_objects, n_objects))
  np.fill_diagonal(diagonal_cooccurrence, 1)
  d = {'method': 'diagonal 1',
       'prefix': 'diagonal',
       'cooccurrence': diagonal_cooccurrence}
  np.save(os.path.join(cooccurrence_path,
                       'cooccurrence.diagonal.npy'), d)

  print('DONE! bye.')


if __name__ == '__main__':

  test_get_nears()
