# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import cache

import numpy as np

CAP_V4L2 = 200
CAP_DSHOW = 700
CAP_AVFOUNDATION = 1200
CAP_ANY = -1

CAP_PROP_FPS = 5
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
COLOR_RGB2BGR = 4
COLOR_BGR2RGB = 4

ROTATE_90_COUNTERCLOCKWISE = 2
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1


@cache
def _generate_image(width: int, height: int):
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def cvtColor(color_image, color_conversion):  # noqa: N802
    if color_conversion in [COLOR_RGB2BGR, COLOR_BGR2RGB]:
        return color_image[:, :, [2, 1, 0]]
    else:
        raise NotImplementedError(color_conversion)


def rotate(color_image, rotation):
    if rotation is None:
        return color_image
    elif rotation == ROTATE_90_CLOCKWISE:
        return np.rot90(color_image, k=1)
    elif rotation == ROTATE_180:
        return np.rot90(color_image, k=2)
    elif rotation == ROTATE_90_COUNTERCLOCKWISE:
        return np.rot90(color_image, k=3)
    else:
        raise NotImplementedError(rotation)


class VideoCapture:
    def __init__(self, index):
        self.index = index
        self.width = 640
        self.height = 480
        self.fps = 30
        self.is_opened = True
        
    def read(self):
        # Generate a random image as mock data
        img = _generate_image(width=self.width, height=self.height)
        return True, img
        
    def isOpened(self):
        return self.is_opened
        
    def release(self):
        self.is_opened = False
        
    def set(self, prop, value):
        if prop == CAP_PROP_FRAME_WIDTH:
            self.width = value
            return True
        elif prop == CAP_PROP_FRAME_HEIGHT:
            self.height = value
            return True
        elif prop == CAP_PROP_FPS:
            self.fps = value
            return True
        return False
        
    def get(self, prop):
        if prop == CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif prop == CAP_PROP_FPS:
            return self.fps
        return 0

# Mock rotation functions
def rotate(img, angle):
    return img

def flip(img, code):
    return img
