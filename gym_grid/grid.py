import numpy as np
from gym_grid.rendering import *
import json

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 1

# Map of color names to RGB values
COLORS = {
    'red'    : np.array([255, 0, 0]),
    'green'  : np.array([0, 200, 0]),
    'blue'   : np.array([0, 0, 255]),
    'purple' : np.array([112, 39, 195]),
    'yellow' : np.array([255, 205, 0]),
    'yellow1': np.array([255, 240, 182]),
    'grey'   : np.array([100, 100, 100]),
    'black'  : np.array([0, 0, 0])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'black' : 6
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'start'         : 1,
    'wall'          : 2,
    'empty'         : 4,
    'goal'          : 5,
    'agent'         : 6,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'goal':
            v = Goal(color)
        elif obj_type == 'start':
            v = Start(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

class Goal(WorldObj):
    def __init__(self, color='green'):
        super().__init__('goal', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Start(WorldObj):
    def __init__(self, color='red'):
        super().__init__('start', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Seen(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('seen', color)
        
    def can_overlap(self):
        return True
    
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class SimpleGrid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = SimpleGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = SimpleGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        tile_seen,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """
        # Hash map lookup key for the cache
        key = (tile_seen, agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8) + 255

        if obj != None:
            obj.render(img)

        if tile_seen and obj == None:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['yellow1'])

        if agent_dir is not None:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS['yellow'])
            
        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.01, 0, 1), COLORS['grey'])
        fill_coords(img, point_in_rect(0, 1, 0, 0.01), COLORS['grey'])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_path=None,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                tile_seen = False
                if (i, j) in agent_path.values():
                    tile_seen = True

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = SimpleGrid.render_tile(
                    cell,
                    tile_seen=tile_seen,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    # def encode(self, vis_mask=None):
    #     """
    #     Produce a compact numpy encoding of the grid
    #     """

    #     if vis_mask is None:
    #         vis_mask = np.ones((self.width, self.height), dtype=bool)

    #     array = np.zeros((self.width, self.height, 3), dtype='uint8')
    #     for i in range(self.width):
    #         for j in range(self.height):
    #             if vis_mask[i, j]:
    #                 v = self.get(i, j)

    #                 if v is None:
    #                     array[i, j, 0] = OBJECT_TO_IDX['empty']
    #                     array[i, j, 1] = 0
    #                     array[i, j, 2] = 0

    #                 else:
    #                     array[i, j, :] = v.encode()

    #     return array

    # @staticmethod
    # def decode(array):
    #     """
    #     Decode an array grid encoding back into a grid
    #     """

    #     width, height, channels = array.shape
    #     assert channels == 3

    #     vis_mask = np.ones(shape=(width, height), dtype=bool)

    #     grid = SimpleGrid(width, height)
    #     for i in range(width):
    #         for j in range(height):
    #             type_idx, color_idx, state = array[i, j]
    #             v = WorldObj.decode(type_idx, color_idx, state)
    #             grid.set(i, j, v)
    #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

    #     return grid, vis_mask

    # def process_vis(grid, agent_pos):
    #     mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

    #     mask[agent_pos[0], agent_pos[1]] = True

    #     for j in reversed(range(0, grid.height)):
    #         for i in range(0, grid.width-1):
    #             if not mask[i, j]:
    #                 continue

    #             cell = grid.get(i, j)
    #             if cell and not cell.see_behind():
    #                 continue

    #             mask[i+1, j] = True
    #             if j > 0:
    #                 mask[i+1, j-1] = True
    #                 mask[i, j-1] = True

    #         for i in reversed(range(1, grid.width)):
    #             if not mask[i, j]:
    #                 continue

    #             cell = grid.get(i, j)
    #             if cell and not cell.see_behind():
    #                 continue

    #             mask[i-1, j] = True
    #             if j > 0:
    #                 mask[i-1, j-1] = True
    #                 mask[i, j-1] = True

    #     for j in range(0, grid.height):
    #         for i in range(0, grid.width):
    #             if not mask[i, j]:
    #                 grid.set(i, j, None)

    #     return mask