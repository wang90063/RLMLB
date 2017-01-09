import pygame
import sys
import random
import numpy as np
from pygame.locals import *
from colour import Color
from collections import Counter
from math import exp

# frame params
FPS = 30  # frames per second, the general speed of the program
CELL_SIZE = 200  # size of cell height & width in pixels
GAP_SIZE = 1  # size of gap between cells in pixels
CELL_COLUMN = 2  # number of columns of cells
CELL_ROW = 1  # number of rows of cells
NUM_CELL = CELL_COLUMN * CELL_ROW  # num of cells

WINDOW_WIDTH = int(CELL_COLUMN * (CELL_SIZE + GAP_SIZE))  # size of window's width in pixels
WINDOW_HEIGHT = int(CELL_ROW * (CELL_SIZE + GAP_SIZE))  # size of windows' height in pixels

NUM_USER = 10
white = Color("white")
COLOR_LIST = list(white.range_to(Color("black"), 20))
RESOURCE_LIST = [18, 2, 6, 10, 8, 6, 6, 10, 7, 10, 5, 7, 4, 8, 9, 5, 6, 5, 9]

# RGB
GRAY = (100, 100, 100)
NAVY_BLUE = (60, 60, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 128, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)
BG_COLOR = NAVY_BLUE
CELL_COLOR = WHITE

pygame.init()
FPSCLOCK = pygame.time.Clock()
DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('MLB System Model')


class SystemModel:
    def __init__(self, ):
        self.users = init_users()
        self.cells = init_cells()

    def frame_step(self, input_action):
        pygame.event.pump()
        # update PRB according to actions
        # for i in range(NUM_CELL):
        #     # take the PRB from cell j to i
        #     cell_to_take = input_action[i]
        #     self.cells[i][3] += 1
        #     self.cells[cell_to_take][3] -= 1
        if input_action[1] == 1:
            if self.cells[0][3] < 20:
                self.cells[0][3] += 1
            if self.cells[1][3] > 1:
                self.cells[1][3] -= 1
        elif input_action[2] == 1:
            if self.cells[1][3] < 20:
                self.cells[1][3] += 1
            if self.cells[0][3] > 1:
                self.cells[0][3] -= 1
        print(self.cells[:, 3])
        # get reward
        reward = get_reward(self.cells)

        # update system states
        self.users = move_user(self.users)
        self.cells = update_load(self.users, self.cells)

        # draw frame
        DISPLAY_SURF.fill(BG_COLOR)
        outrage_ratio = [x[4] / x[3] for x in self.cells]
        outrage_ratio = [min(x, 1) for x in outrage_ratio]  # larger than 1 is outrage, use black color directly
        color_index = [int(x * len(COLOR_LIST)) for x in outrage_ratio]
        for cell in self.cells:
            this_color_index = color_index[cell[2]]
            try:
                this_cell_color = [i * 255 for i in list(COLOR_LIST[this_color_index - 1].rgb)]
            except:
                print("this_color_index: ", this_color_index)
                print("COLOR_LIST length: ", len(COLOR_LIST))
            pygame.draw.rect(DISPLAY_SURF, this_cell_color, (cell[0], cell[1], CELL_SIZE, CELL_SIZE))

        # draw users
        for user in self.users:
            pygame.draw.circle(DISPLAY_SURF, RED, (user[0], user[1]), 2)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward


def init_users():
    """
    initialize user. every user consists of 4 params:
    (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
    while medium and high takes 20% and 10%.
    :return: user: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
     """
    user_x = np.random.randint(WINDOW_WIDTH, size=NUM_USER, dtype='int')
    user_y = np.random.randint(WINDOW_HEIGHT, size=NUM_USER, dtype='int')
    cell_id = which_cell(loc_x=user_x, loc_y=user_y)
    mobility_type = np.random.choice([1, 5, 10], size=NUM_USER, p=[0.7, 0.2, 0.1])  # low(70%), medium(20%), high(10%)
    users = np.vstack((user_x, user_y, cell_id, mobility_type))
    return users.T


def which_cell(loc_x, loc_y):
    """
    calculate which cell the user is in
    :param loc_x:
    :param loc_y:
    :return: cell_id
    """
    column = np.ceil(loc_x / CELL_SIZE)
    row = np.ceil(loc_y / CELL_SIZE)
    cell_id = (row - 1) * CELL_COLUMN + column
    return cell_id.astype(int)


def init_cells():
    """
    initialize cell list, every cell in the lists consists of 5 params:
    (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
    :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
    """
    # cell location
    flatten_x = np.tile(np.arange(CELL_COLUMN), CELL_ROW)
    flatten_y = np.repeat(np.arange(CELL_ROW), CELL_COLUMN)
    cell_x = flatten_x * (CELL_SIZE + GAP_SIZE)
    cell_y = flatten_y * (CELL_SIZE + GAP_SIZE)

    cell_id = np.arange(NUM_CELL)
    cell_PRB = np.array(RESOURCE_LIST[:NUM_CELL])
    cell_load = np.zeros(NUM_CELL)

    cells = np.vstack((cell_x, cell_y, cell_id, cell_PRB, cell_load))
    return cells.T.astype(int)


def move_user(users):
    """
    user mobility func update users' location in every frame. mobility range comes from user mobility type. Meanwhile,
    user should only move in the cell range, restricted by the MARGIN.
    """
    mobility = users[:, 3]

    move_x = [random.randint(-x, x) for x in mobility]
    user_x = users[:, 0] + move_x  # update loc according to user mobility type
    users[:, 0] = np.clip(user_x, 4, WINDOW_WIDTH- 4)  # restrict user loc in the cell range

    move_y = [random.randint(-x, x) for x in mobility]
    user_y = users[:, 1] + move_y  # update loc according to user mobility type
    users[:, 1] = np.clip(user_y, 4, WINDOW_HEIGHT - 4)  # restrict user loc in the cell range

    cell_id = which_cell(users[:, 0], users[:, 1])
    users[:, 2] = cell_id
    return users


def update_load(users, cells):
    """
    calculate cell load according to the sum of users in its range.
    """
    # count users in each cell
    user_in = users[:, 2] - 1
    user_count = Counter(user_in).most_common()
    # update the load of each cell in cell list
    for item in user_count:
        cells[item[0]][4] = item[1]
    return cells


def get_reward(cells):
    resource = cells[:, 3]
    load = cells[:, 4]
    # normal = (resource > load).astype(int)
    # reward = np.sum(normal - 1)
    gap = np.clip(resource - load, a_min=-10000, a_max=0)
    absmax = np.max(abs(gap)) + exp(1e-5)
    gap = np.divide(gap, absmax)
    reward = np.sum(gap)
    return reward


def main():
    system_model = SystemModel()
    while True:
        items = range(NUM_CELL)
        random_action = random.sample(items, len(items))
        imagedata, reward = system_model.frame_step(random_action)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    main()
