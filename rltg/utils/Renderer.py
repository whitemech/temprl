from abc import abstractmethod
from time import sleep

import cv2


class Renderer(object):
    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

class PixelRenderer(Renderer):

    def __init__(self, width=600, height=600, window_name='obs', delay=1, skip_frame=1, video=False):
        self.window_name = window_name
        self.delay = delay
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        self.video = video
        if video:
            self.vid = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*"XVID"), float(30), (160, 210), False)

        self.counter = 0
        self.skip_frame = skip_frame


    def update(self, env):
        screen = env.render()
        self.counter += 1
        if self.counter%self.skip_frame==0:
            cv2.imshow(self.window_name, screen)
            cv2.waitKey(self.delay)
            if self.video:
                self.vid.write(screen)

    def release(self):
        if self.video:
            self.vid.release()

class PygameRenderer(Renderer):
    def __init__(self, delay=0.1, skip_frame=1):
        self.counter = 0
        self.skip_frame = skip_frame
        self.delay = delay

    def update(self, env):
        self.counter += 1
        if self.counter % self.skip_frame == 0:
            env.render()
            sleep(self.delay)

