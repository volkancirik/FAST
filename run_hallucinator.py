import paths
import torch
import base64
from pprint import pprint
from tqdm import tqdm
import os

import sys
import numpy as np
from refer360_utils import get_object_dictionaries
from refer360_utils import load_butd, load_vectors, DIR2IDX
import csv
DIRECTIONS = {
    'navigation': ['ul', 'u', 'ur', 'l', 'r', 'dl', 'd', 'dr'],
    'canonical': ['up', 'down', 'left', 'right'],
    'cartesian': ['vertical', 'horizontal'],
    'lup': ['lateral', 'up', 'down'],
    'canonical_proximity': ['close_up', 'close_down', 'close_left', 'close_right',
                            'far_up', 'far_down', 'far_left', 'far_right']
}

file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(
    file_path, '..', '..', 'build_refer360'))
sys.path.append(module_path)
csv.field_size_limit(sys.maxsize)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dump_predictions(model, nodes, features_prefix, fov_prefix, object_classes):

  n_objects = len(object_classes)-1
  all_preds = np.zeros((len(nodes), n_objects, len(DIR2IDX)))

  for jj, n in enumerate(sorted(nodes.keys())):
    idx = nodes[n]['idx']
    fov_file = fov_prefix + '.{}.jpg'.format(idx)
    img_feat_path = fov_file.replace('.jpg', '.feat.npy')

    preds = np.zeros((n_objects, len(DIR2IDX)))
    if os.path.exists(img_feat_path):
      img_features = np.load(img_feat_path)
      image = torch.FloatTensor(img_features).to(DEVICE)
      img_batch = image.unsqueeze(0).repeat(n_objects, 1, 1, 1)

      query = []
      for obj_id, obj_name in enumerate(object_classes[:n_objects]):
        query += [obj_id]

      latitude = torch.FloatTensor([nodes[n]['lat']]).to(
          DEVICE).unsqueeze(0).repeat(n_objects, 1)
      longitude = torch.FloatTensor([nodes[n]['lng']]).to(
          DEVICE).unsqueeze(0).repeat(n_objects, 1)

      query = torch.LongTensor([query]).to(DEVICE).transpose(1, 0)
      out = model({
          'latitude': latitude,
          'longitude': longitude,
          'im_batch': img_batch,
          'queries': query,
      })
      preds = out['action_logits'].cpu().detach().numpy()  # 359 x 8
    all_preds[idx, :, :] = preds
  return all_preds


def run_hallucination(
        cache_root='/projects2/refer360test/data/cached_data_30degrees',
        butd_filename='./img_features/refer360_30degrees_obj36.tsv',
        image_list_file='./refer360_data/imagelist.txt',
        n_fovs=60,
        word_embedding_path='./tasks/FAST/data/cc.en.300.vec',
        obj_dict_file='./tasks/FAST/data/vg_object_dictionaries.all.json',
        model_path='/projects2/refer360test/src/exp-PRIOR/default/model.best.pt',
        outfile='img_features/refer360_30degrees_hallucinated_v4.tsv'):

  image_list = [line.strip()
                for line in open(image_list_file)]

  print('\nloading model from', model_path)
  model = torch.load(model_path)
  model.eval()

  meta_file = os.path.join(cache_root, 'meta.npy')
  meta = np.load(meta_file, allow_pickle=True)[()]
  nodes = meta['nodes']

  vg2idx, idx2vg, obj_classes, name2vg, name2idx, vg2name = get_object_dictionaries(
      obj_dict_file, return_all=True)
  n_objects = len(vg2name)-1
  print('# of objects (except unk)', n_objects)

  print('loading w2v...', word_embedding_path)
  w2v = load_vectors(word_embedding_path, name2vg)

  print('loading BUTD boxes...', butd_filename)
  fov2keys = load_butd(butd_filename,
                       vg2name=vg2name,
                       keys=['boxes', 'objects_id'])
  print('loaded BUTD boxes!', image_list_file)

  fov2keys = {'boxes': {}, 'objects_id': {}}

  FIELDNAMES = ['pano_fov', 'features']

  print('Generating hallucinated object directions')
  pbar = tqdm(image_list)

  with open(outfile, 'w') as tsvfile:
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)

    for fname in pbar:
      pano = fname.split('/')[-1].split('.')[0]
      features_prefix = os.path.join(
          cache_root, 'features', '{}'.format(pano))
      fov_prefix = os.path.join(
          cache_root, 'fovs', '{}'.format(pano))
      preds = dump_predictions(
          model, nodes, features_prefix, fov_prefix, obj_classes)  # n_fovs x n_obj x dir

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

        for direction in DIR2IDX:
          feat_index = DIR2IDX[direction]
          pred_index = DIRECTIONS['navigation'].index(direction)
          dir_feats = np.zeros((1, 300), dtype=np.float32)

          for co_idx in range(n_objects):
            w = preds[idx, co_idx, pred_index]
            if w > 0:
              co_name = obj_classes[co_idx]
              emb = w2v.get(co_name, w2v['</s>'])
              dir_feats += emb * w
          features[feat_index, :] = dir_feats

        encoded = base64.b64encode(features).decode()
        d = {'pano_fov': pano_fov,
             'features': encoded}
        writer.writerow(d)


if __name__ == '__main__':
  run_hallucination()
