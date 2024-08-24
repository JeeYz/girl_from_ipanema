import pygame
import os
import sys


class AppMain(object):
    def __init__(self, basic_setting_builder) -> None:

        # pygame status
        self._is_running = False

        # pygame app variables
        self._size = basic_setting_builder._size
        self._background_color = basic_setting_builder._background_color
        self._fps_num = basic_setting_builder._fps_time

        # pygame init
        pygame.init()
        self._screen = pygame.display.set_mode(self._size)
        self._clock = pygame.time.Clock()

    # setting boolean
    def set_is_running(self, input_bool):
        self._is_running = input_bool

    def do_execute(self):
        if self._is_running == False:
            self._is_running = True

        while self._is_running:
            self.on_exit()
            self.on_events()
            self.on_loop()
            self.on_render()
            pygame.display.flip()
            self._clock.tick(self._fps_num)

        self.on_clean_up()

    def on_exit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_running = False

    def on_events(self):
        pass

    def on_loop(self):
        pass

    def on_render(self):
        self._screen.fill(self._background_color)

    def on_clean_up(self):
        pygame.quit()

 
