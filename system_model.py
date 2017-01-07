import pygame
import sys
import random
import numpy as np
from pygame.locals import *
from colour import Color
from collections import Counter


# frame params
FPS = 3  # frames per second, the general speed of the program
WINDOW_WIDTH = 1000  # size of window's width in pixels
WINDOW_HEIGHT = 600  # size of windows' height in pixels
CELL_SIZE = 200  # size of cell height & width in pixels
GAP_SIZE = 1  # size of gap between cells in pixels
CELL_COLUMN = 3  # number of columns of cells
CELL_ROW = 2  # number of rows of cells
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
pygame.init()
FPS_CLOCK = pygame.time.Clock()
DISPLAY_SURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('MLB System Model')


class SystemModel:
	def __init__(self):
		self.users = self.init_users()
		self.cells = self.init_cells()

	def init_users(self):
		"""
		initialize user. every user consists of 4 params:
		(1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
		user mobility type is divided into 3 categories: low, medium and high. Low mobility users takes 70% of all,
		while medium and high takes 20% and 10%.
		:return: user: (1) loc_x(center) (2) loc_y(center) (3) which cell user is in (4) user mobility type
		 """
		user_x = np.random.randint(LEFT_MARGIN, RIGHT_MARGIN, size=NUM_USER)
		user_y = np.random.randint(TOP_MARGIN, BOTTOM_MARGIN, size=NUM_USER)
		cell_id = self.which_cell(loc_x=user_x, loc_y=user_y)
		mobility_type = np.random.choice(3, size=NUM_USER, p=[0.7, 0.2, 0.1])  # low(70%), medium(20%), high(10%)
		users = np.vstack((user_x, user_y, cell_id, mobility_type))
		return users.T

	@staticmethod
	def init_cells():
		"""
		initialize cell list, every cell in the lists consists of 5 params:
		(1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
		:return: cell_list: (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
		"""
		# cell location
		flatten_x = np.tile(np.arange(CELL_COLUMN), CELL_ROW)
		flatten_y = np.repeat(np.arange(CELL_ROW), CELL_COLUMN)
		cell_x = flatten_x * (CELL_SIZE + GAP_SIZE) + LEFT_MARGIN
		cell_y = flatten_y * (CELL_SIZE + GAP_SIZE) + TOP_MARGIN

		cell_id = np.arange(NUM_CELL)
		cell_PRB = np.array(RESOURCE_LIST[:NUM_CELL])
		cell_load = np.zeros(NUM_CELL)

		cells = np.vstack((cell_x, cell_y, cell_id, cell_PRB, cell_load))
		return cells.T

	@staticmethod
	def which_cell(loc_x, loc_y):
		"""
		calculate which cell the user is in
		:param loc_x:
		:param loc_y:
		:return: cell_id
		"""
		column = np.ceil((loc_x - LEFT_MARGIN) / CELL_SIZE)
		row = np.ceil((loc_y - TOP_MARGIN) / CELL_SIZE)
		cell_id = (row - 1) * CELL_COLUMN + column
		return cell_id

	def move_user(self):
		"""
		user mobility func update users' location in every frame. mobility range comes from user mobility type. Meanwhile,
		user should only move in the cell range, restricted by the MARGIN.
		"""
		mobility = self.users[:, 3]

		move_x = mobility * np.random.uniform(-1, 1, size=len(mobility))
		user_x = self.users[:, 0] + move_x  # update loc according to user mobility type
		self.users[:, 0] = np.clip(user_x, LEFT_MARGIN + 4, RIGHT_MARGIN - 4)  # restrict user loc in the cell range

		move_y = mobility * np.random.uniform(-1, 1, size=len(mobility))
		user_y = self.users[:, 1] + move_y  # update loc according to user mobility type
		self.users[:, 1] = np.clip(user_y, TOP_MARGIN + 4, BOTTOM_MARGIN - 4)  # restrict user loc in the cell range

	def update_load(self):
		"""
		calculate cell load according to the sum of users in its range.
		"""
		# count users in each cell
		user_in = self.users[:, 2] - 1
		user_count = Counter(user_in).most_common()
		# update the load of each cell in cell list
		for item in user_count:
			print(item[0])
			self.cells[item[0]][4] = item[1]

	def get_reward(self):
		resource = self.cells[:, 3]
		load = self.cells[:, 4]
		normal = resource > load
		reward = np.sum(normal.astype(int) - 1)
		return reward

	def draw_cells(self):
		"""
		draw cell square and paint color according to cell location and outrage ratio. outrage ratio denotes the balance
		between cell resources and cell load. Color in each cell is chosen according to the outrage ratio. Dark color means
		the cell is high-burden, opposite otherwise. Black color means ratio is lager than 1, meaning that cell is outrage.
		:param cell_list: (1)loc_x(left) (2)loc_y(top) (3)NO. (4)PRB number (5)load
		:return:
		"""
		outrage_ratio = [x[4] / x[3] for x in self.cells]
		# print(cell_list)
		# print(outrage_ratio)
		outrage_ratio = [min(x, 1) for x in outrage_ratio]  # larger than 1 is outrage, use black color directly
		# print_list = [round(x, 2) for x in outrage_ratio]
		# print(print_list)
		color_index = [int(x * len(COLOR_LIST)) for x in outrage_ratio]
		for cell in self.cells:
			this_color_index = color_index[int(cell[2] - 1)]
			this_cell_color = [i * 255 for i in list(COLOR_LIST[this_color_index - 1].rgb)]
			pygame.draw.rect(DISPLAY_SURF, this_cell_color, (cell[0], cell[1], CELL_SIZE, CELL_SIZE))

	def draw_users(self):
		"""
		draw user circle according to their center postition
		:param user_list: (1) loc_x(center) (2) loc_y(center)
		:return:
		"""
		for user in self.users:
			pygame.draw.circle(DISPLAY_SURF, RED, (int(user[0]), int(user[1])), 2)

	def frame_step(self, input_action):
		# update PRB according to actions
		for i in range(NUM_CELL):
			# take the PRB from cell j to i
			cell_to_take = input_action[i]
			self.cells[i][3] += 1
			self.cells[cell_to_take][3] -= 1
		print(self.cells)

		# update system states
		self.move_user()
		self.update_load()
		reward = self.get_reward()

		# draw frame
		DISPLAY_SURF.fill(BG_COLOR)
		self.draw_cells()
		self.draw_users()
		return reward

def main():
	system_model = SystemModel()
	while True:
		items = range(6)
		random_action = random.sample(items, len(items))
		# random_action = [0,0,0,0,0,0]
		reward = system_model.frame_step(random_action)
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
