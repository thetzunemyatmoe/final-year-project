import pygame
import numpy as np
import json

class GridWorldGUI:
    def __init__(self, grid_width=20, grid_height=15, cell_size=20):
        pygame.init()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.grid = np.zeros((grid_height, grid_width), dtype=int)

        # Initialize outer cells as obstacles
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[grid_height - 1, :] = 1
        self.grid[:, grid_width - 1] = 1

        self.screen_width = grid_width * cell_size
        self.screen_height = grid_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Grid World Editor")
        self.running = True
        self.dragging = False
        self.toggle_to = None

        self.main_loop()

    def draw_grid(self):
        self.screen.fill((255, 255, 255))
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                color = (0, 0, 0) if self.grid[i, j] == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, 
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (200, 200, 200), 
                                 (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size), 1)
        pygame.display.flip()

    def toggle_cell(self, row, col):
        if 0 < row < self.grid_height-1 and 0 < col < self.grid_width-1:
            self.grid[row, col] = self.toggle_to

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.resize_screen(event.w, event.h)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.dragging = True
                    pos = pygame.mouse.get_pos()
                    row = pos[1] // self.cell_size
                    col = pos[0] // self.cell_size
                    if 0 < row < self.grid_height-1 and 0 < col < self.grid_width-1:
                        self.toggle_to = 1 if self.grid[row, col] == 0 else 0
                        self.toggle_cell(row, col)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False
                    self.toggle_to = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Save grid
                    self.save_grid('grid_world.json')
                elif event.key == pygame.K_l:  # Load grid
                    self.load_grid('grid_world.json')

        if self.dragging:
            pos = pygame.mouse.get_pos()
            row = pos[1] // self.cell_size
            col = pos[0] // self.cell_size
            self.toggle_cell(row, col)

    def resize_screen(self, new_width, new_height):
        self.cell_size = min(new_width // self.grid_width, new_height // self.grid_height)
        self.screen_width = self.grid_width * self.cell_size
        self.screen_height = self.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

    def main_loop(self):
        while self.running:
            self.handle_events()
            self.draw_grid()

        pygame.quit()

    def get_grid(self):
        return self.grid

    def save_grid(self, filename):
        with open(filename, 'w') as f:
            # Flip the grid along the x-axis for consistency
            self.grid = np.flipud(self.grid)
            json.dump(self.grid.tolist(), f)

    def load_grid(self, filename):
        try:
            with open(filename, 'r') as f:
                self.grid = np.array(json.load(f))
                # Flip the grid along the x-axis for consistency
                self.grid = np.flipud(self.grid)
                self.grid_height, self.grid_width = self.grid.shape
                self.resize_screen(self.screen_width, self.screen_height)
        except FileNotFoundError:
            print(f"No such file: '{filename}'")

# Usage
if __name__ == "__main__":
    gui = GridWorldGUI(grid_width = 30, grid_height= 30)
    grid_world = gui.get_grid()
    gui.save_grid('grid_world.json')
    #print(grid_world)