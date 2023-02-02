import pathlib

import pygame

# Size of the square tiles used in this environment.
TILE_SIZE = 32

# Grid
ROWS = 16
COLS = 16

NUM_TILES = ROWS * COLS
NUM_ACTIONS = 4
INITIAL_STATE = 0

# Resolution to emulate
VIRTUAL_WIDTH = TILE_SIZE * COLS
VIRTUAL_HEIGHT = TILE_SIZE * ROWS

# Scale factor between virtual screen and window
H_SCALE = 2
V_SCALE = 2

# Resolution of the actual window
WINDOW_WIDTH = VIRTUAL_WIDTH * H_SCALE
WINDOW_HEIGHT = VIRTUAL_HEIGHT * V_SCALE

# Battery
BATTERY_LOAD = 31
BATTERY_WIDTH = TILE_SIZE
BATTERY_HEIGHT = TILE_SIZE / 2
BATTERY_LEFT = VIRTUAL_WIDTH - BATTERY_WIDTH - 10
BATTERY_TOP = 10

# Default pause time between steps (in seconds)
DEFAULT_DELAY = 0.5

BASE_DIR = pathlib.Path(__file__).parent

# Textures used in the environment
TEXTURES = {
    'floor': pygame.image.load(BASE_DIR / "assets" / "graphics" / "floor.png"),
    'goal': pygame.image.load(BASE_DIR / "assets" / "graphics" / "goal.png"),
    'character': [
        pygame.image.load(BASE_DIR / "assets" / "graphics" / "panda_left.png"),
        pygame.image.load(BASE_DIR / "assets" / "graphics" / "panda_down.png"),
        pygame.image.load(BASE_DIR / "assets" / "graphics" / "panda_right.png"),
        pygame.image.load(BASE_DIR / "assets" / "graphics" / "panda_up.png"),
    ]
}

# Initializing the mixer
pygame.mixer.init()

# Loading music
pygame.mixer.music.load(BASE_DIR / "assets" / "sounds" / "main_theme.ogg")

# Sound effects
SOUNDS = {
    'win': pygame.mixer.Sound(BASE_DIR / "assets" / "sounds" / "won.wav")
}

# Initializing fonts
pygame.font.init()

# Fonts
FONTS = {
    'font': pygame.font.Font(BASE_DIR /"assets"/"fonts"/"font.ttf", 16)
}