import pygame
import sys
import random
import math
import numpy as np
from pygame.locals import *
from colour import Color


FPS = 3  # frames per second, the general speed of the program
WINDOW_WIDTH = 1000  # size of window's width in pixels
WINDOW_HEIGHT = 600  # size of windows' height in pixels
CELL_SIZE = 200  # size of cell height & width in pixels
GAP_SIZE = 1  # size of gap between cells in pixels
CELL_COLUMN = 2  # number of columns of cells
CELL_ROW = 3  # number of rows of cells
NUM_CELL = CELL_COLUMN * CELL_ROW  # num of cells
LEFT_MARGIN = int((WINDOW_WIDTH - (CELL_COLUMN * (CELL_SIZE + GAP_SIZE))) / 2)
RIGHT_MARGIN = int(WINDOW_WIDTH - LEFT_MARGIN)
TOP_MARGIN = int((WINDOW_HEIGHT - (CELL_ROW * (CELL_SIZE + GAP_SIZE))) / 2)
BOTTOM_MARGIN = int(WINDOW_HEIGHT - TOP_MARGIN)
NUM_USER = 20
white = Color("white")
COLOR_LIST = list(white.range_to(Color("black"), 15))
RESOURCE_LIST = [10, 8, 6, 10, 8, 6, 6, 10, 7, 10, 5, 7, 4, 8, 9, 5, 6, 5, 9]

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


def init_users():
    """
    initialize user lists. every user in the lists consists of 4 params:
    (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
    while medium and high takes 20% and 10%.
    :return: user_list: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    """
    user_list = list()
    for i in range(1, NUM_USER):
        user = list()
        user.append(random.randint(LEFT_MARGIN, RIGHT_MARGIN))  # user x_axis position
        user.append(random.randint(TOP_MARGIN, BOTTOM_MARGIN))  # user y_axis position
        user.append(which_cell(user[0], user[1]))  # denote user is in which cell
        # user mobility type
        random_num = random.randint(0, 10)
        if random_num <= 5:  # low(70%): 0-1 per sec
            user.append(1)
        elif random_num <= 7:  # medium(20%): 0-5 per sec
            user.append(1)
        else:
            user.append(1)  # high(10%): 0-20 per sec
        user_list.append(user)
    return user_list


def init_cells():
    """
    initialize cell list, every cell in the lists consists of 5 params:
    (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
    :return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
    """
    cell_list = list()
    for row in range(CELL_ROW):
        for column in range(CELL_COLUMN):
            cell = list()
            cell_x = column * (CELL_SIZE + GAP_SIZE) + LEFT_MARGIN  # cell axis_x
            cell_y = row * (CELL_SIZE + GAP_SIZE) + TOP_MARGIN  # cell axis_y
            cell_id = row * CELL_COLUMN + column + 1  # cell serial number
            cell_load = 0  # cell load initialized with 0
            cell.extend((cell_x, cell_y, cell_id, RESOURCE_LIST[cell_id-1], cell_load))
            cell_list.append(cell)
    return cell_list


def user_mobility(user_list):
    """
    user mobility func update users' location in every frame. mobility range comes from user mobility type. Meanwhile,
    user should only move in the cell range, restricted by the MARGIN.
    :param user_list: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    :return: new_user_list: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
    """
    new_user_list = list()
    for user in user_list:
        #  update loc according to user mobility type
        ii = random.randint(-user[3], user[3])
        print("user[3]= ", user[3])
        print("ii= ", ii)
        user[0] += random.randint(-user[3], user[3])
        user[1] += random.randint(-user[3], user[3])
        #  restrict user loc in the cell range
        user[0] = max(user[0], LEFT_MARGIN + 4)
        user[0] = min(user[0], RIGHT_MARGIN - 4)
        user[1] = max(user[1], TOP_MARGIN + 4)
        user[1] = min(user[1], BOTTOM_MARGIN - 4)
        #  update which cell user is in
        user[2] = which_cell(user[0], user[1])
        new_user_list.append(user)
    return new_user_list


def draw_cells(cell_list):
    """
    draw cell square and paint color according to cell location and outrage ratio. outrage ratio denotes the balance
    between cell resources and cell load. Color in each cell is chosen according to the outrage ratio. Dark color means
    the cell is high-burden, opposite otherwise. Black color means ratio is lager than 1, meaning that cell is outrage.
    :param cell_list: (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
    :return:
    """
    outrage_ratio = [x[4]/x[3] for x in cell_list]
    # print(cell_list)
    # print(outrage_ratio)
    outrage_ratio = [min(x, 1) for x in outrage_ratio]  # larger than 1 is outrage, use black color directly
    # print_list = [round(x, 2) for x in outrage_ratio]
    # print(print_list)
    color_index = [int(x * len(COLOR_LIST)) for x in outrage_ratio]

    for cell in cell_list:
        this_color_index = color_index[cell[2]-1]
        this_cell_color = [i * 255 for i in list(COLOR_LIST[this_color_index-1].rgb)]
        pygame.draw.rect(DISPLAY_SURF, this_cell_color, (cell[0], cell[1], CELL_SIZE, CELL_SIZE))


def which_cell(loc_x, loc_y):
    """
    calculate which cell the user is in
    :param loc_x:
    :param loc_y:
    :return: cell_id
    """
    column = int(math.ceil((loc_x - LEFT_MARGIN) / CELL_SIZE))
    row = int(math.ceil((loc_y - TOP_MARGIN) / CELL_SIZE))
    cell_id = (row - 1) * CELL_COLUMN + column
    return cell_id


def draw_users(user_list):
    """
    draw user circle according to their center postition
    :param user_list: (1) loc_x(center) (2) loc_y(center)
    :return:
    """
    for user in user_list:
        pygame.draw.circle(DISPLAY_SURF, RED, (user[0], user[1]), 2)


def cal_cell_load(cell_list, user_list):
    """
    calculate cell load according to the sum of users in its range.
    :param cell_list:
    :param user_list:
    :return:
    """
    # count users in each cell
    cell_load = [0] * CELL_COLUMN * CELL_ROW
    for user in user_list:
        cell_load[user[2] - 1] += 1
    # print(cell_load)

    # update the load of each cell in cell list
    for i in range(len(cell_list)):
        cell_list[i][4] = cell_load[i]
    return cell_list


def cal_reward(cell_list):
    reward = 0
    for i in range(NUM_CELL):
        resource = cell_list[i][3]
        load = cell_list[i][4]
        if resource < load:
            reward -= 1
    return reward


def frame_step(input_action, cell_list, user_list):
    # update PRB according to actions
    for i in range(NUM_CELL):
        # take the PRB from cell j to i
        cell_to_take = input_action[i]
        cell_list[i][3] += 1
        cell_list[cell_to_take][3] -= 1
    print(cell_list)

    # update system states
    user_list = user_mobility(user_list)
    cell_list = cal_cell_load(cell_list, user_list)
    reward = cal_reward(cell_list)

    # draw frame
    DISPLAY_SURF.fill(BG_COLOR)
    draw_cells(cell_list)
    draw_users(user_list)
    return reward


def main():
    global FPS_CLOCK
    global DISPLAY_SURF
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    user_list = init_users()
    cell_list = init_cells()
    while True:
        # DISPLAY_SURF.fill(BG_COLOR)
        # user_list = user_mobility(user_list)
        # cell_list = cal_cell_load(cell_list, user_list)
        # reward = cal_reward(cell_list)
        # draw_cells(cell_list)
        # draw_users(user_list)
        items = range(6)
        random_action = random.sample(items, len(items))
        # random_action = [0,0,0,0,0,0]
        print(random_action)
        reward = frame_step(random_action, cell_list, user_list)
        print(reward)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
        # Redraw the screen and wait a clock tick.
        pygame.display.update()
        FPS_CLOCK.tick(FPS)


if __name__ == '__main__':
    main()
