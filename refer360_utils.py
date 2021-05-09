''' Utils for Refer360 grounding environment '''
from collections import defaultdict
import numpy as np
import base64
import io
import json
from tqdm import tqdm
import os
import sys
import csv
from nltk.corpus import wordnet
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

  print('loading cnn features with prefix:', feature_prefix)
  pbar = tqdm(image_list)
  print('cached features root:', cache_root)
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
        item[key] = item[key].reshape(shape)
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
  np.save('cooccurrence.test.npy', d)
  d = np.load('cooccurrence.test.npy', allow_pickle=True)[()]
  cooccurrence = d['cooccurrence']
  method = d['method']
  print(method)
  print(cooccurrence[idx1, :])


def get_refer360_stats(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=240,
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json'):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])

  n_objects = len(vg2name)
  cooccurrence = np.zeros((n_objects, n_objects))
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
  d = {'method': 'refer360_30degrees_butd_36obj',
       'prefix': 'r30butd_v3',
       'butd_filename': butd_filename,
       'cooccurrence': cooccurrence}
  np.save('cooccurrence.r30butd_v3.npy', d)
  print('DONE! bye.')


def get_spatialsense_stats(
        spatialsense_annotations='/projects3/all_data/spatialsense/annotations.json',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json'):

  anns = json.load(open(spatialsense_annotations, 'r'))

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  cooccurrence = np.zeros((n_objects, n_objects))
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

  d = {'method': 'spatialsense',
       'prefix': 'ss_v3',
       'cooccurrence': cooccurrence}
  np.save('cooccurrence.ss_v3.npy', d)
  print('DONE! bye.')


def get_visualgenome_stats(
        visualgenome_objects='/projects3/all_data/visualgenome/objects.json',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json'):

  objects = json.load(open(visualgenome_objects, 'r'))

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)

  print('{}x{} cooccurrence matrix will be created...'.format(n_objects, n_objects))
  cooccurrence = np.zeros((n_objects, n_objects))
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

  d = {'method': 'visualgenome v1.4',
       'prefix': 'vg_v3',
       'cooccurrence': cooccurrence}
  np.save('cooccurrence.vg_v3.npy', d)
  print('DONE! bye.')


def wordnet_similarity(word1, word2):
  synsets1 = wordnet.synsets(word1)
  synsets2 = wordnet.synsets(word2)
  if synsets1 == [] or synsets2 == []:
    return 0
  wordFromList1 = synsets1[0]
  wordFromList2 = synsets2[0]
  return wordFromList1.wup_similarity(wordFromList2)


def get_wordnet_stats(obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json'):

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
       'prefix': 'wn_v3',
       'cooccurrence': cooccurrence}
  np.save('cooccurrence.wn_v3.npy', d)
  print('DONE! bye.')


def dump_fov_caches(
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=240,
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        cooccurrence_file='/projects1/Matterport3DSimulator/cooccurrence.vg.npy'):

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)

  cooccurrence_data = np.load(cooccurrence_file,
                              allow_pickle=True)[()]
  cooccurrence = cooccurrence_data['cooccurrence']
  # normalize the counts
  for idx in range(cooccurrence.shape[0]):
    sum_count = np.sum(cooccurrence[idx, :])
    if sum_count > 0:
      cooccurrence[idx, :] = cooccurrence[idx, :] / sum_count
    else:
      cooccurrence[idx, :] = 1.0 / cooccurrence.shape[0]

  prefix = cooccurrence_data['prefix']
  outfile = 'img_features/refer360_30degrees_{}.tsv'.format(prefix)
  print('output file:', outfile)

  image_list = [line.strip()
                for line in open(image_list_file)]
  pbar = tqdm(image_list)

  FIELDNAMES = ['pano_fov', 'features']
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
          for ii, box in enumerate(boxes):
            directions = []
            if box[1] < 120:
              directions.append('u')
            elif box[1] > 280:
              directions.append('d')
            if box[0] < 120:
              if directions:
                directions[0] += 'l'
              directions.append('l')
            elif box[0] > 280:
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
        d = {'pano_fov': pano_fov,
             'features': encoded}
        writer.writerow(d)

  print('DONE! bye.')


if __name__ == '__main__':
  # test_get_nears()
  get_refer360_stats()
  get_visualgenome_stats()
  get_spatialsense_stats()
  get_wordnet_stats()
  cooccurrence_files = [
      '/projects1/Matterport3DSimulator/cooccurrence.vg_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.wn_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.ctrl_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.xlm_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.gpt3_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.gpt2_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.gpt_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.gpt3_vg_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.ss_v3.npy',
      '/projects1/Matterport3DSimulator/cooccurrence.r30butd_v3.npy',
  ]
  for cooccurrence_file in cooccurrence_files:
    dump_fov_caches(cooccurrence_file=cooccurrence_file)
