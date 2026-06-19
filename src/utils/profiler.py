import time

class Profiler:
    """
    Simple profiling utility to track FPS and processing latency.
    """
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.stats = {}
        self.timers = {}
        self.last_frame_time = time.time()
        self.fps = 0.0

    def start(self, name):
        self.timers[name] = time.time()

    def stop(self, name):
        if name in self.timers:
            latency = time.time() - self.timers[name]
            if name not in self.stats:
                self.stats[name] = latency
            else:
                self.stats[name] = (self.stats[name] * (1 - self.smoothing)) + (latency * self.smoothing)

    def tick(self):
        """Called once per frame to calculate FPS."""
        now = time.time()
        frame_time = now - self.last_frame_time
        self.last_frame_time = now
        
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        if self.fps == 0.0:
            self.fps = current_fps
        else:
            self.fps = (self.fps * (1 - self.smoothing)) + (current_fps * self.smoothing)

    def get_latency_ms(self, name):
        return self.stats.get(name, 0.0) * 1000.0

    def get_fps(self):
        return self.fps
