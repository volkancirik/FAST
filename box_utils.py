import numpy as np
import os
import sys

file_path = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(file_path))
sys.path.append(module_path)
module_path = os.path.abspath(os.path.join(
    file_path, '..', '..', 'build_refer360'))
sys.path.append(module_path)


def get_intersection(box1, box2):
  left = box1[0] if box1[0] > box2[0] else box2[0]
  top = box1[1] if box1[1] > box2[1] else box2[1]
  right = box1[2] if box1[2] < box2[2] else box2[2]
  bottom = box1[3] if box1[3] < box2[3] else box2[3]
  if left > right or top > bottom:
    return [0, 0, 0, 0]
  return [left, top, right, bottom]


def calculate_area(box):
  return (box[2] - box[0]) * (box[3] - box[1])


def calculate_iou(box1, box2):
  area1 = calculate_area(box1)
  area2 = calculate_area(box2)
  intersection = get_intersection(box1, box2)
  area_int = calculate_area(intersection)
  return area_int / (area1 + area2 - area_int)


def get_distance(x0, y0, x1, y1):
  return ((x0-x1)**2 + (y0-y1)**2)**0.5


def get_line_distance(x0, y0, x1, y1, x, y):
  term0 = (y1-y0)*x
  term1 = -(x1-x0)*y
  term2 = x1*y0
  term3 = -y1*x0
  term4 = (y1-y0)**2
  term5 = (x1-x0)**2
  term6 = abs(term0 + term1 + term2 + term3)
  term7 = (term4 + term5) ** 0.5
  return term6/term7


def get_boxes2coor_relationships(boxes, coor):
  y = coor[0]
  x = coor[1]
  inside = [0]*len(boxes)
  center_proximity = [0.]*len(boxes)
  edge_proximity = [0.]*len(boxes)
  edge_type = [0]*len(boxes)

  id2edge = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
  for ii, b in enumerate(boxes):
    x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
    if x0 <= x <= x1 and y0 <= y <= y1:
      inside[ii] = 1
    center_proximity[ii] = get_distance((x0+x1)/2, (y0+y1)/2, x, y)
    edges = []
    # up
    edges.append(get_line_distance(x0, y0, x1, y0, x, y))
    # right
    edges.append(get_line_distance(x1, y0, x1, y1, x, y))
    # down
    edges.append(get_line_distance(x0, y1, x1, y1, x, y))
    # left
    edges.append(get_line_distance(x0, y0, x0, y1, x, y))
    edge_proximity[ii] = np.min(edges)
    edge_type[ii] = id2edge[np.argmin(edges)]
  return inside, center_proximity, edge_proximity, edge_type


def get_box2box_relationships(box1, box2):

  x10, y10, x11, y11 = box1[0], box1[1], box1[2], box1[3]
  x20, y20, x21, y21 = box2[0], box2[1], box2[2], box2[3]

  edges1_v = [
      [x10, y10, x10, y11],  # left v
      [x11, y10, x11, y11]  # right v
  ]
  edges1_h = [
      [x10, y10, x11, y10],  # top h
      [x10, y11, x11, y11],  # bottom h
  ]
  edges2_v = [
      [x20, y20, x20, y21],
      [x21, y20, x21, y21]
  ]
  edges2_h = [
      [x20, y20, x21, y20],
      [x20, y21, x21, y21],
  ]
  vedges = []
  vedgenames = []
  for ii, v1 in enumerate(edges1_v):
    for jj, v2 in enumerate(edges2_v):
      x, y = v1[0], (v1[1] + v1[3])/2

      if x == v2[0]:  # on same vertical line
        if v1[1] <= v2[1] <= v1[3]:
          distance = 0
        elif v1[1] <= v2[3] <= v1[3]:
          distance = 0
        elif v2[1] <= v1[1] <= v2[3]:
          distance = 0
        elif v2[1] <= v1[3] <= v2[3]:
          distance = 0
        else:
          distance = float('inf')
      else:
        distance = get_line_distance(v2[0], v2[1], v2[2], v2[3], x, y)
      vedges.append(distance)
      vedgenames.append('v{}->{}'.format(ii, jj))

  hedges = []
  hedgenames = []

  for ii, h1 in enumerate(edges1_h):
    for jj, h2 in enumerate(edges2_h):
      x, y = (h1[0]+h1[2]) / 2, h1[1]

      if y == h2[1]:  # on same vertical line
        if h1[0] <= h2[0] <= h1[2]:
          distance = 0
        elif h1[0] <= h2[2] <= h1[2]:
          distance = 0
        elif h2[0] <= h1[0] <= h2[2]:
          distance = 0
        elif h2[0] <= h1[2] <= h2[2]:
          distance = 0
        else:
          distance = float('inf')
      else:
        distance = get_line_distance(h2[0], h2[1], h2[2], h2[3], x, y)
      hedges.append(distance)
      hedgenames.append('h{}->{}'.format(ii, jj))
  vedge_proximity = np.min(vedges)
  vedge_id = np.argmin(vedges)
  hedge_proximity = np.min(hedges)
  hedge_id = np.argmin(hedges)

  edge_proximity = vedge_proximity + hedge_proximity
  edge_name = vedgenames[vedge_id] + hedgenames[hedge_id]
  return edge_proximity, edge_name


def test_box2box():
  b1 = [5, 5, 50, 50]
  b2 = [60, 40, 100, 100]
  e_p, e_id = get_box2box_relationships(b1, b2)
  print(e_p, e_id)

  b1 = [10, 20, 100, 25]
  b2 = [20, 26, 100, 50]
  e_p, e_id = get_box2box_relationships(b1, b2)
  print(e_p, e_id)


if __name__ == '__main__':
  test_box2box()
