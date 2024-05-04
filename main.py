from scipy.spatial import distance

class Road:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.init_porperties()

    def init_properties(self):
        self.length = distance.euclidean(self.start, self.end)
        self.angle_sin = (self.end[1]-self.start[1]) / self.length
        self.angle_cos = (self.end[0]-self.start[0]) / self.length









def __init__(self, config={}):
  # Set default configuration
  self.set_default_config()

  # Update configuration
  for attr, val in config.items():
      setattr(self, attr, val)