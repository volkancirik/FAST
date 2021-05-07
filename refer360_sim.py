from panoramic_camera_cached import CachedPanoramicCamera
from collections import namedtuple
WorldState = namedtuple(
    "WorldState", ["scanId", "viewpointId", "heading", "elevation", "viewIndex"])


class Refer360Simulator(CachedPanoramicCamera):
  def __init__(self, cache_root,
               fov=90,
               output_image_shape=(400, 400),
               height=2276,
               width=4552):
    super(Refer360Simulator, self).__init__(
        cache_root, fov, output_image_shape, height, width)

  def newEpisode(self, world_state):
    self.set_pano(world_state.scanId)
    self.look_fov(world_state.viewpointId)

  def getState(self):
    world_state = WorldState(self.pano, self.idx, self.lng, self.lat, 4)
    return world_state
