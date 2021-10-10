from panoramic_camera_cached import CachedPanoramicCamera
from collections import namedtuple
WorldState = namedtuple(
    'WorldState', ['scanId', 'viewpointId', 'heading', 'elevation', 'viewIndex', 'img'])
ReadingWorldState = namedtuple(
    'ReadingWorldState', ['scanId', 'viewpointId', 'heading', 'elevation', 'viewIndex', 'img', 'sentId'])


class Refer360Simulator(CachedPanoramicCamera):
  def __init__(self, cache_root,
               fov=90,
               output_image_shape=(400, 400),
               height=2276,
               width=4552,
               reading=False,
               raw=False):

    super(Refer360Simulator, self).__init__(
        cache_root, fov, output_image_shape, height, width)
    self.reading = reading
    self.sentId = 0
    self.raw = raw

  def newEpisode(self, world_state):
    self.set_pano(world_state.scanId)
    self.look_fov(world_state.viewpointId)
    if self.reading:
      self.sentId = world_state.sentId

  def getState(self):
    img = None

    if self.raw:
      img = self.get_image()
    if self.reading:
      return ReadingWorldState(self.pano, self.idx, self.lng, self.lat, 4, img, self.sentId)
    return WorldState(self.pano, self.idx, self.lng, self.lat, 4, img)
