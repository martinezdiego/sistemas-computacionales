import pygame

from .. import settings


class DrawableMixin:
    def render(self, surface):
        if isinstance(self.texture_name, tuple):
            texture_name = self.texture_name[0]
            frame_index = self.texture_name[1]
        else:
            texture_name = self.texture_name
            frame_index = self.frame_index
        texture = settings.GAME_TEXTURES[texture_name]
        frame = settings.GAME_FRAMES[texture_name][frame_index]
        image = pygame.Surface((frame.width, frame.height), pygame.SRCALPHA)
        image.fill((0, 0, 0, 0))
        image.blit(texture, (0, 0), frame)
        surface.blit(image, (self.x, self.y))
