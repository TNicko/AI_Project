import gym
from gym import spaces, utils
import numpy as np
import shutil
from io import StringIO
from contextlib import closing
from typing import Optional
from gym_grid.grid import SimpleGrid, Wall, Goal, Start
from gym.envs.toy_text.utils import categorical_sample
from gym_grid.window import Window

TILE_SIZE = 100

MAPS = {
    "5x5": ["SEEEE", "EEEEE", "EEEEE", "EEEEE", "EEEGE"],
}

REWARD_MAP = {
        b'E': 0.0,
        b'S': 0.0,
        b'W': -1.0,
        b'G': 1.0,
    }

class GridEnv(gym.Env):

    def __init__(self, desc: list[str] =None, map_name: str =None, reward_map: dict[bytes, float] =None):
        """
        Parameters
        ----------
        desc: list[str]
            Custom map for the environment.
        map_name: str
            ID to use any of the preloaded maps.
        reward_map: dict[bytes, float]
            Custom reward map.
        """

        self.desc = self.__initialise_desc(desc, map_name)
        self.nrow, self.ncol = self.desc.shape
        
        # Initialise action and state spaces
        self.nA = 4
        self.nS = self.nrow * self.ncol
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # Reward
        self.reward_map = self.__initialise_reward_map(reward_map)
        self.reward_range = (min(self.reward_map.values()), max(self.reward_map.values()))

        # Initialise env dynamics
        self.initial_state = None
        self.initial_state_distrib = self.__get_initial_state_distribution(self.desc)
        self.P = self.__get_env_dynamics()

        # Rendering
        self.window = None
        self.grid = self.__initialise_grid_from_desc(self.desc)
        self.fps = 3

    @staticmethod
    def __initialise_desc(desc: list[str], map_name: str) -> np.ndarray:

        if desc is not None:
            return np.asarray(desc, dtype="c")
        # if desc is not None and map_name is None:
        #     desc = generate_random_map()
            return np.asarray(desc, dtype="c")
        if desc is None and map_name is not None:
            desc = MAPS[map_name]
            return np.asarray(desc, dtype="c")

    @staticmethod
    def __initialise_grid_from_desc(desc: list[str]) -> SimpleGrid:
        """
        Initialise the grid to be rendered from the desc matrix.
        @NOTE: the point (x,y) in the desc matrix corresponds to the
        point (y,x) in the rendered matrix.
        Parameters
        ----------
        desc: list[list[str]]
            Custom map for the environment.
        
        Returns
        -------
        grid: SimpleGrid
            The grid to be rendered.
        """
        nrow, ncol = desc.shape
        grid = SimpleGrid(width=ncol, height=nrow)
        for row in range(nrow):
            for col in range(ncol):
                letter = desc[row, col]
                if letter == b'G':
                    grid.set(col, row, Goal())
                elif letter == b'W':
                    grid.set(col, row, Wall(color='black'))
                else:
                    grid.set(col, row, None)
        return grid

    @staticmethod
    def __initialise_reward_map(reward_map: dict[bytes, float]) -> dict[bytes, float]:
        if reward_map is None:
            return REWARD_MAP
        else:
            return reward_map

    @staticmethod
    def __get_initial_state_distribution(desc: list[str]) -> np.ndarray:
        """
        Get the initial state distribution.
        
        If desc contains multiple times the letter 'S', then the initial
        state distribution will a uniform on the respective states and the
        initial state radomly sampled from it.     
        Parameters
        ----------
        desc: list[str]
            Custom map for the environment.
        
        Returns:
        --------
        initial_state_distrib: np.ndarray
        Examples
        --------
        >>> desc = ["SES", "WEE", "SEG"]
        >>> SimpleGridEnv.__get_initial_state_distribution(desc)
        array([0.33333333, 0.        , 0.33333333, 0.        , 0.        ,
        0.        , 0.33333333, 0.        , 0.        ])
        """

        initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        initial_state_distrib /= initial_state_distrib.sum()
        return initial_state_distrib

    def __to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def __to_next_xy(self, row: int, col: int, a: int) -> tuple[int, int]:
        """
        Compute the next position on the grid when starting at (row, col)
        and taking the action a.
        Remember:
        - 0: LEFT
        - 1: DOWN
        - 2: RIGHT
        - 3: UP
        """
        if a == 0:
            col = max(col - 1, 0)
        elif a == 1:
            row = min(row + 1, self.nrow - 1)
        elif a == 2:
            col = min(col + 1, self.ncol - 1)
        elif a == 3:
            row = max(row - 1, 0)

        return (row, col)

    def __transition(self, row: int, col: int, a: int) -> tuple[int, float, bool, bool]:
        """
        Compute next state, reward and done when starting at (row, col)
        and taking the action action a.
        """
        newrow, newcol = self.__to_next_xy(row, col, a)
        newstate = self.__to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        terminated = bytes(newletter) in b"GW"
        truncated = False
        reward = self.reward_map[newletter]

        return newstate, reward, terminated, truncated
    
    def __get_env_dynamics(self):
        """
        Compute the dynamics of the environment.
        For each state-action-pair, the following tuple is computed:
            - the probability of transitioning to the next state
            - the next state
            - the reward
            - the done flag
        """

        nrow, ncol = self.nrow, self.ncol
        nA, nS = self.nA, self.nS

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nrow):
            for col in range(ncol):
                s = self.__to_s(row, col)
                for a in range(nA):
                    li = P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GW":
                        li.append((1.0, s, 0, True)) #@NOTE: is reward=0 correct? Probably the value doesn't matter.
                    else:
                        li.append((1.0, *self.__transition(row, col, a)))
        return P

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        return_info: bool = True,
        options: Optional[dict] = None,
        ):
            super().reset(seed=seed)
            # sample initial state from the initial state distribution
            self.s = categorical_sample(self.initial_state_distrib, self.np_random)
            # set the starting red tile on the grid to render

            if self.initial_state is not None:
                self.grid.set(self.initial_state % self.ncol, self.initial_state // self.ncol, None)
            self.grid.set(self.s % self.ncol, self.s // self.ncol, Start())

            self.initial_state = self.s
            self.lastaction = None

            if not return_info:
                return int(self.s)
            else:
                return int(self.s), {"prob": 1}

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, term, trunc = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, term, trunc, {"prob": p})

    def render(self, mode="human"):
        if mode == "ansi":
            return self.__render_text(self.desc.tolist())
        elif mode == "human":
            return self.__render_gui()
        elif mode == "rgb_array":
            return self.__render_rgb_array()
        else:
            raise ValueError(f"Unsupported rendering mode {mode}")
    
    def __render_gui(self):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) to the grid.render method.
        """
        img = self.grid.render(
            tile_size=TILE_SIZE,
            agent_pos=(self.s % self.ncol, self.s // self.ncol),
            agent_dir=0
        )
        if not self.window:
            self.window = Window('my_custom_env')
            self.window.show(block=False)
        self.window.show_img(img, self.fps)

    def __render_rgb_array(self):
        """
        Render the environment to an rgb array.
        """
        img = self.grid.render(
            tile_size=TILE_SIZE,
            agent_pos=(self.s % self.ncol, self.s // self.ncol),
            agent_dir=0
        )
        print(img)
        return img

    def __render_text(self, desc):
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(F"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            print(outfile.getvalue())
            return outfile.getvalue()

    def close(self):
        if self.window:
            self.window.close()
        return

    def generate_random_map(size=8, p=0.8):
        """
        Generates a random valid map (one that has a path from start to goal)
        
        Parameters
        ----------
        size: int 
            Size of each side of the grid
        p: float
            Probability that a tile is empty
        """
        pass

        