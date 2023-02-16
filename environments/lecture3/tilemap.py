import settings


class Tile:
    def __init__(self, x, y, texture_name):
        self.x = x
        self.y = y
        self.texture_name = texture_name

    def render(self, surface):
        surface.blit(settings.TEXTURES[self.texture_name], (self.x, self.y))


class TileMap:
    def __init__(self, num_rows, num_cols, tile_texture_names):
        self.tiles = []
        tile_counter = 0
        for i in range(num_rows):
            for j in range(num_cols):
                self.tiles.append(
                    Tile(
                        j * settings.TILE_SIZE,
                        i * settings.TILE_SIZE,
                        tile_texture_names[tile_counter],
                    )
                )
                tile_counter += 1

    def render(self, surface):
        for tile in self.tiles:
            tile.render(surface)
