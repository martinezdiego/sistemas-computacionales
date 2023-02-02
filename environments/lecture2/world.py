import pygame

import settings
from tilemap import TileMap

class World:
    def __init__(self, title, state, action, finish_state=None):
        pygame.init()
        pygame.display.init()
        pygame.mixer.music.play(loops=-1)
        self.render_surface = pygame.Surface(
            (settings.VIRTUAL_WIDTH, settings.VIRTUAL_HEIGHT)
        )
        self.screen = pygame.display.set_mode(
            (settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT)
        )
        pygame.display.set_caption(title)
        self.state = state
        self.action = action
        self.render_character = True
        self.render_goal = True
        self.tilemap = None
        self.finish_state = finish_state
        self.__create_tilemap()

    def __create_tilemap(self):
        tile_texture_names = ["floor" for _ in range(settings.NUM_TILES)]
        self.tilemap = TileMap(tile_texture_names)

    def reset(self, state, action):
        self.state = state
        self.action = action
        self.render_character = True
        self.render_goal = True

    def update(self, state, action, reward, terminated):
        if state == self.finish_state:
            self.render_goal = False
            settings.SOUNDS['win'].play()

        self.state = state
        self.action = action

    def render(self, current_battery, score):
        self.render_surface.fill((0, 0, 0))

        self.tilemap.render(self.render_surface)
        
        text_obj = settings.FONTS['font'].render("Power", True, (0, 0, 0))
        text_rect = text_obj.get_rect()
        text_rect.center = (settings.BATTERY_LEFT + settings.BATTERY_WIDTH * 0.5 , settings.BATTERY_HEIGHT * 0.5)
        self.render_surface.blit(text_obj, text_rect)

        pygame.draw.rect(self.render_surface, (0,0,0), (settings.BATTERY_LEFT, settings.BATTERY_HEIGHT, settings.BATTERY_WIDTH, settings.BATTERY_HEIGHT))
        pygame.draw.rect(self.render_surface, (91,194,54), (settings.BATTERY_LEFT, settings.BATTERY_HEIGHT, settings.BATTERY_WIDTH * current_battery, settings.BATTERY_HEIGHT))

        # render score
        text_obj = settings.FONTS['font'].render("Win  Lose", True, (0, 0, 0))
        text_rect = text_obj.get_rect()
        text_rect.center = (settings.VIRTUAL_WIDTH * 0.5 , settings.BATTERY_HEIGHT * 0.5)
        self.render_surface.blit(text_obj, text_rect)

        text_obj = settings.FONTS['font'].render(f"{score[0]}  {score[1]}", True, (0, 0, 0))
        text_rect = text_obj.get_rect()
        text_rect.center = (settings.VIRTUAL_WIDTH * 0.5 , settings.BATTERY_HEIGHT + settings.BATTERY_HEIGHT * 0.5)
        self.render_surface.blit(text_obj, text_rect)

        if self.render_goal:
            self.render_surface.blit(
                settings.TEXTURES['goal'],
                (self.tilemap.tiles[self.finish_state].x,
                 self.tilemap.tiles[self.finish_state].y)
            )

        if self.render_character:
            self.render_surface.blit(
                settings.TEXTURES['character'][self.action],
                (self.tilemap.tiles[self.state].x,
                 self.tilemap.tiles[self.state].y)
            )

        self.screen.blit(
            pygame.transform.scale(
                self.render_surface,
                self.screen.get_size()),
            (0, 0)
        )

        pygame.event.pump()
        pygame.display.update()

    def close(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.display.quit()
        pygame.quit()


    
