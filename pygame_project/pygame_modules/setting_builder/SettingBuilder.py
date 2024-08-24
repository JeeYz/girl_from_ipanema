

from pygame_modules.AppMain import AppMain


class BasicSettingBuilder:
    def __init__(self) -> None:
        self._basic_width = None
        self._basic_height = None
        self._size = None
        self._background_color = None
        self._fps_time = None

    def set_basic_width(self, input_width):
        self._basic_width = input_width
        return self

    def set_basic_height(self, input_height):
        self._basic_height = input_height
        return self
        
    def set_size(self):
        self._size = self._basic_width, self._basic_height
        return self

    def set_background_color(self, input_color):
        self._background_color = input_color
        return self

    def set_fps_time(self, input_fps):
        self._fps_time = input_fps
        return self

    def set_up_basic(self):
        return AppMain(self)
