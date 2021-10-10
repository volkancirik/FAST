from tqdm import tqdm
import numpy as np
from PIL import Image
import os


def cache_r2r_scans():
  print('will R2R panoramaas for the simulator.')
  scans_root = './data/v1/scans/'
  pbar = tqdm(os.listdir(scans_root))

  for scan in pbar:
    viewpoints_root = os.path.join(scans_root, scan, 'panos_small')
    viewpoint2idx = {}
    pano_list = []

    sim_cache_file = os.path.join(scans_root, scan, 'sim_cache.npy')
    if os.path.exists(sim_cache_file):
      continue

    for idx, viewpoint in enumerate(os.listdir(viewpoints_root)):
      pano_file = os.path.join(scans_root, scan, 'panos_small', viewpoint)

      pano = np.expand_dims(np.array(Image.open(pano_file)), axis=0)
      pano_list += [pano]

      viewpoint2idx[viewpoint.split('.')[0]] = idx

    panos = np.concatenate(pano_list, axis=0)
    data = {
        'panos': panos,
        'viewpoint2idx': viewpoint2idx
    }
    np.save(sim_cache_file, data)
    pbar.set_description(scan)


def test_r2r_scans():
  scans_root = './data/v1/scans/'
  pbar = tqdm(os.listdir(scans_root))

  for scan in pbar:
    sim_cache = os.path.join(scans_root, scan, 'sim_cache.npy')
    if not os.path.exists(sim_cache):
      print('{} does not exist! generate cache file!'.format(sim_cache))
      quit(0)
  print('all good!')


if __name__ == '__main__':
  cache_r2r_scans()
  test_r2r_scans()
