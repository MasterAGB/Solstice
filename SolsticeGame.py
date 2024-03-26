import json
import random
import pygame
import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
import pyperclip

from GameSettings import GameSettings


class SolsticeGame:
    # Define each tile type's index in the channel dimension
    used_channel_indices = {}

    def __init__(self, level_index=1, game_skin="default", device="cpu"):
        global win, win_size;

        self.device = device
        self.level_name = None
        self.skins = ['default', 'portal', 'bombs', 'forest', 'ice', 'castle', 'last']

        self.skin = game_skin
        self.last_action = None
        self.SetupLevel(level_index)

        # Define action mapping: 0=Left, 1=Down, 2=Right, 3=Up
        # self.action_space = np.arange(4)
        # self.observation_space = np.arange(len(self.map_layout) * len(self.map_layout[0]))
        self.enableRendering = True

        self.action_size = 4  # Assuming Left, Down, Right, Up

        pygame.init()
        # Load the music file
        pygame.mixer.music.load("tiles/Audio.mp3")
        # Play the music

        self.game_settings = GameSettings()

        # Example: toggling music based on settings
        if self.game_settings.get_setting("music_enabled"):
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

        win_size = (737, 744)
        win = pygame.display.set_mode(win_size)
        pygame.display.set_caption("Solstice Play")

    def ToggleMusic(self):

        current_status = self.game_settings.get_setting("music_enabled")
        new_status = not current_status
        self.game_settings.set_setting("music_enabled", new_status)

        # Apply the new setting
        if new_status:
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

    def action_space_sample(self):
        # Returns a random action from 0, 1, 2, 3
        return random.randint(0, 3)

    def generate_solvable_map(self, rows, cols):
        # Initialize map with 'F' (free) cells
        game_map = [['.' for _ in range(cols)] for _ in range(rows)]

        # Starting point
        game_map[0][0] = 'S'

        # Ensure 'G' is placed at the bottom-right corner
        game_map[-1][-1] = 'G'

        # Create a guaranteed solvable path from S to G
        cur_row, cur_col = 0, 0

        # Randomly add holes, ensuring 'G' remains at its location
        for _ in range(int(rows * cols * 0.6)):  # Adjust density of 'H' as needed
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            if game_map[r][c] == '.':
                game_map[r][c] = 'H'

        while cur_row < rows - 1 or cur_col < cols - 1:
            if cur_row < rows - 1 and (random.choice([True, False]) or cur_col == cols - 1):
                cur_row += 1  # Move down
            else:
                cur_col += 1  # Move right
            if cur_row < rows and cur_col < cols:
                game_map[cur_row][cur_col] = 'U'  # Mark path as 'F'

        # Reaffirm 'G' placement in case of overwrites
        game_map[-1][-1] = 'G'

        return ["".join(row) for row in game_map]

    def load_map_layout(self, level_index):

        # Construct the file name based on the level identifier
        file_name = f"levels/level_{level_index}.json"

        try:
            # Open the level file and load its content
            with open(file_name, 'r') as file:
                level = json.load(file)
                self.level_name = level['name']  # Store level name
                self.skin = level['style']  # Set skin based on level
                return level['map_structure']
        except FileNotFoundError:
            print(f"Level file {file_name} not found - WIN the game.")
            return self.generate_solvable_map(21, 21)  # Fallback to a default map
        except json.JSONDecodeError:
            print(f"Error reading level file {file_name}.")
            return self.generate_solvable_map(21, 21)  # Fallback to a default map


    def PrintLevel(self):
        # Create a dictionary with the level data
        level_data = {
            "name": self.level_name,
            "style": self.skin,
            "map_structure": self.map_layout
        }

        # Convert the dictionary to a JSON-formatted string
        level_json = json.dumps(level_data, indent=2)

        # Print to console
        print(level_json)

        # Copy to clipboard
        pyperclip.copy(level_json)
        print("Level JSON has been copied to the clipboard.")

        # Use Tkinter's simpledialog to show the JSON in a copy-able popup
        # This requires a Tkinter root window, which we create and then hide
        root = tk.Tk()
        root.withdraw()  # We don't want a full GUI, so keep the root window from appearing
        simpledialog.messagebox.showinfo("Level JSON", level_json)  # Show the JSON in a messagebox

        # Destroy the Tkinter root window when done
        root.destroy()

    def step(self, action):
        self.step_count = self.step_count + 1;

        if self.is_dizzy:
            # With some probability, choose a random action instead of the intended one
            if random.random() < 0.33:  # Adjust this probability as needed
                action = self.action_space_sample()

        # Simplified step function with directional movement.
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        reward = 0
        is_terminated = False
        is_truncated = False  # TMP removed
        info = {}  # Placeholder for additional info.

        # Calculate current position
        row = self.state // cols
        col = self.state % cols
        row_prev = row
        col_prev = col

        self.last_action = action
        # Determine new position based on action
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(rows - 1, row + 1)
        elif action == 2:  # Right
            col = min(cols - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)

        if self.map_layout[row][col] in ['W', 'T', 'C']:
            row = row_prev
            col = col_prev

        if(self.key_count==0):
            if self.map_layout[row][col] in ['A']:
                row = row_prev
                col = col_prev

        # Update state
        self.state = row * cols + col


        # Check for game termination conditions
        cell_prev = self.map_layout[row_prev][col_prev]
        if cell_prev == 'U':
            self.replaceThisCell(row_prev, col_prev, "H")

        cell = self.map_layout[row][col]
        if cell == 'G':
            is_terminated = True
            reward = 1
        elif cell == 'H':
            is_terminated = True

        # terminating if i stepped on monster
        elif cell == 'M':
            is_terminated = True
        elif cell == 'F':
            is_terminated = True
        elif cell == 'Q':
            is_terminated = True

        elif cell == 'D':
            if (self.has_antidot == False):
                self.is_dizzy = True
            self.replaceThisCell(row, col, ".")
        elif cell == 'A':
            self.key_count-=1;
            self.replaceThisCell(row, col, "L")
        elif cell == 'P':
            reward = 0.3
            self.has_antidot = True
            self.is_dizzy = False
            self.replaceThisCell(row, col, ".")
        elif cell == 'K':
            reward = 0.3
            self.replaceThisCell(row, col, ".")
            self.replaceAllCells("C", "G") #Opening the Gate
        elif cell == 'R':
            reward = 0.3
            self.replaceAllCells("T", "J") #Opening all chests with Keys
        elif cell == 'J':
            self.key_count+=1;
            reward = 0.3
            self.replaceThisCell(row, col, ".")
            #not doing this: self.replaceAllCells("A", "L") #Opening the Door
        elif cell == 'B':
            reward = 0.3
            self.replaceThisCell(row, col, ".")
            self.replaceAllCells("M", ".") #killing monster
            self.replaceAllCells("F", "K") #replacing F Monster with a goal key
            self.replaceAllCells("Q", "J") #replacing KeyMonster with a door key



        self.moveAllMobs("M", ".")
        self.moveAllMobs("F", ".")
        self.moveAllMobs("Q", ".")

        #terminating also if monster has stepped on me
        cell = self.map_layout[row][col]
        if cell == 'M':
            is_terminated = True
        elif cell == 'F':
            is_terminated = True
        elif cell == 'Q':
            is_terminated = True


        if (self.step_count > 20000):
            is_truncated = True;

        self.render();
        state_tensor = self.generate_multi_channel_state();

        return self.state, state_tensor, reward, is_terminated, is_truncated, info

    selected_tile_type = '.'
    def handle_mouse_click(self, x, y):
        print(f"Clicked on {x}x{y}")

        # Check if click is in the tile selection area
        for tile_type, _, position in self.tile_selection:
            tile_rect = pygame.Rect(position[0], position[1], self.tile_selection_w, self.tile_selection_h)  # Assuming tile size
            if tile_rect.collidepoint(x, y):
                self.selected_tile_type = tile_type  # Save selected tile type
                print(f"selected_tile_type on {self.selected_tile_type}")
                return

        # Convert screen coordinates to isometric grid coordinates
        # This is a simplified conversion and may need adjustment
        grid_x, grid_y = self.screen_to_iso(x, y)

        print(f"grid_x on {grid_x}x{grid_y}")

        if 0 <= grid_x < len(self.map_layout[0]) and 0 <= grid_y < len(self.map_layout):
            # Replace the tile at the calculated grid position
            self.map_layout[grid_y] = (
                    self.map_layout[grid_y][:grid_x] + self.selected_tile_type +
                    self.map_layout[grid_y][grid_x + 1:]
            )

    cursor_grid_x=-1;
    cursor_grid_y=-1;
    def handle_mouse_move(self, x, y):
        print(f"Moved on {x}x{y}")
        # Convert screen coordinates to isometric grid coordinates
        # This is a simplified conversion and may need adjustment
        self.cursor_grid_x, self.cursor_grid_y = self.screen_to_iso(x, y)

        print(f"moved grid_x on {self.cursor_grid_x}x{self.cursor_grid_y}")


    def screen_to_iso(self, screen_x, screen_y):
        global win, win_size
        # This conversion assumes a direct mapping and needs to be adjusted based on your isometric projection
        tile_width = 32
        tile_height = 16
        tile_width_half = tile_width/2   # Half the width of your tile
        tile_height_half = tile_height/2   # Half the height of your tile

        x = screen_x - win_size[0]/2;
        y = screen_y - win_size[1]/2 - tile_height * 3  + 160; #  +
        grid_x = (x / tile_width_half + y / tile_height_half)/2/2
        grid_y = (y / tile_height_half - (x / tile_width_half))/2/2

        # Calculate offset needed to center the character
        offset_grid_row = 0
        offset_grid_col = 0

        grid_size_rows = len(self.map_layout)
        grid_size_cols = len(self.map_layout[0])
        if (grid_size_rows > 9 or grid_size_cols > 9):
            current_row = self.state // grid_size_cols
            current_col = self.state % grid_size_cols

            offset_grid_row = - current_row
            offset_grid_col = - current_col

            # Max tiles on screen
            viewport_tiles_x = 11
            viewport_tiles_y = 15

            # Maximum allowable offsets given the level size - 4 and 2 - magic numbers
            max_offset_x = grid_size_cols - viewport_tiles_x // 2 - 4 + 4
            max_offset_y = grid_size_rows - viewport_tiles_y // 2 - 2 + 4

            # Calculate desired offset to center the character
            desired_offset_col = -current_col
            desired_offset_row = -current_row

            # Clamp the offsets to the maximum allowable values to avoid showing too much empty space
            offset_grid_col = max(desired_offset_col, -max_offset_x)
            offset_grid_row = max(desired_offset_row, -max_offset_y)


        return int(grid_x-offset_grid_col), int(grid_y-offset_grid_row)


    tile_selection = []  # Clear previous selections
    tile_selection_w = 32;
    tile_selection_h = 32;

    def draw_tile_selection_area(self, tiles_per_row=11):
        # Define starting position for drawing tiles
        start_x, start_y = 30, 604
        tile_spacing = 20  # Space between tiles
        row_spacing = 8  # Vertical space between rows
        outline_thickness = 2  # Thickness of the selection outline

        # Tile types and their display names for the editor
        tile_types = {
            'S': 'start', '.': 'free', 'H': 'hole', 'G': 'goal', 'C': 'goalClosed',
            'K': 'key', 'J': 'key2O', 'T': 'key2L', 'R': 'press', 'L': 'door',
            'A': 'doorClosed', 'Q': 'mob3', 'M': 'mob', 'F': 'mob2', 'U': 'unstable',
            'D': 'dizzy', 'P': 'potion', 'B': 'bomb', 'W': 'wall'
        }

        def scale_and_crop_image(image, target_width, target_height):
            # Calculate the scale factor to match the target width
            original_width, original_height = image.get_size()
            scale_factor = target_width / original_width
            scaled_height = int(original_height * scale_factor)

            # Scale the image
            scaled_image = pygame.transform.scale(image, (target_width, scaled_height))

            # Crop the bottom part of the scaled image to match the target height
            if scaled_height > target_height:
                # Calculate the area to keep
                crop_rect = pygame.Rect(0, scaled_height - target_height, target_width, target_height)
                cropped_image = scaled_image.subsurface(
                    crop_rect).copy()  # .copy() is necessary to create a new surface
            else:
                cropped_image = scaled_image

            return cropped_image

        # Load and draw each tile type
        for i, (tile_type, image_name) in enumerate(tile_types.items()):
            image = pygame.image.load(f'tiles/{self.skin}/{image_name}.png')
            scaled_image = scale_and_crop_image(image, self.tile_selection_w, self.tile_selection_h)  # Scale for selection area


            # Calculate row and column based on tile index
            row = i // tiles_per_row
            col = i % tiles_per_row

            # Calculate position based on row and column
            x = start_x + col * (tile_spacing + self.tile_selection_w)
            y = start_y + row * (tile_spacing + self.tile_selection_h // 2 + row_spacing)
            position = (x, y)

            pygame.draw.rect(win, (0, 0, 0), (
                x - outline_thickness, y - outline_thickness, self.tile_selection_w + outline_thickness * 2,
                self.tile_selection_h + outline_thickness * 2))

            # Draw tile with a black background and yellow outline if it's the selected one
            if tile_type == self.selected_tile_type:
                pygame.draw.rect(win, (255, 255, 0), (
                x - outline_thickness, y - outline_thickness, self.tile_selection_w + outline_thickness * 2,
                self.tile_selection_h + outline_thickness * 2), outline_thickness)
            win.blit(scaled_image, position)
            self.tile_selection.append((tile_type, scaled_image, position))

    def stepEditor(self, action, event):
        # Simplified step function with directional movement.
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        reward = 0
        is_terminated = False
        is_truncated = False  # TMP removed
        info = {}  # Placeholder for additional info.

        # Calculate current position
        row = self.state // cols
        col = self.state % cols
        row_prev = row
        col_prev = col

        self.last_action = action
        # Determine new position based on action
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(rows - 1, row + 1)
        elif action == 2:  # Right
            col = min(cols - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)


        # Update state
        self.state = row * cols + col

        # Check for game termination conditions
        cell_prev = self.map_layout[row_prev][col_prev]
        cell = self.map_layout[row][col]

        self.renderEditor(event, False);
        state_tensor = self.generate_multi_channel_state();

        return self.state, state_tensor, reward, is_terminated, is_truncated, info

    def NeedToDrawEmptyTileUnder(self, tile):
        return (tile in ['.', 'M', 'F', 'B', 'D', 'P', 'K', 'J', 'T', 'L', 'A', 'Q'])

    def IsUnderTile(self, tile):
        return (tile in ['.', 'S', 'G', 'C', 'H', 'R', 'U'])

    def IsOverTile(self, tile):
        return not self.IsUnderTile(tile)


    def generate_multi_channel_state(self, generateImages=False):
        # Normalize the indices to be continuous starting from 0
        normalized_channel_indices = self.used_channel_indices

        # Correctly increase the number of channels by 1 to include the player's position.
        num_channels = len(normalized_channel_indices) + 1  # This should reflect in state_tensor initialization

        # print(f"Number of channels (including player position): {num_channels}")

        # Initialize the state tensor with zeros, explicitly setting the channel dimension.
        state_tensor = torch.zeros((num_channels, self.level_height, self.level_width))

        # Populate the tensor based on the map layout for predefined channel indices.
        for y, row in enumerate(self.map_layout):
            for x, cell in enumerate(row):
                if cell in normalized_channel_indices:
                    channel = normalized_channel_indices[cell]
                    state_tensor[channel, y, x] = 1

        # Calculate player's current position.
        player_row = self.state // self.level_width
        player_col = self.state % self.level_width

        # IMPORTANT: Verify the last channel is correctly assigned for the player's position.
        state_tensor[num_channels - 1, player_row, player_col] = 1  # Use num_channels - 1 instead of -1

        if generateImages:
            symbol_descriptions = {
                '.': 'Free Space',
                'H': 'Hole',
                'G': 'Goal',
                'K': 'Key',
                'C': 'Closed Goal',
                'F': 'Monster with Key Drop',
                'U': 'Unstable Floor',
                'B': 'Bomb',
                'W': 'Wall',
                'S': 'Start',

                'J': 'Door keys',
                'T': 'Chest with key',
                'R': 'Chest opener',
                'L': 'Opened door',
                'A': 'Closed door',
                'Q': 'Monster -door key',
            }

            i = 0
            # Generate and save BW images for each channel
            for channel, index in normalized_channel_indices.items():
                i += 1;
                plt.figure(figsize=(2, 2))
                plt.imshow(state_tensor[index].cpu().numpy(), cmap='gray', interpolation='none')
                title = f'Ch {index}: {symbol_descriptions[channel]}'  # Use channel symbol as title
                plt.title(title)
                plt.axis('off')
                plt.savefig(f'channels/channel_{index}_{channel}.png')
                plt.close()
                # plt.show()

                self.DrawChannelImage(f'channels/channel_{index}_{channel}.png', i, num_channels);

            i += 1;
            # Generate and show the player position separately if needed
            plt.figure(figsize=(2, 2))
            plt.imshow(state_tensor[num_channels - 1].cpu().numpy(), cmap='gray', interpolation='none')
            plt.title(f'Ch {num_channels - 1}: Player Position')
            plt.axis('off')
            plt.savefig(f'channels/channel_{num_channels - 1}_player_position.png')
            # plt.show()
            plt.close()
            self.DrawChannelImage(f'channels/channel_{num_channels - 1}_player_position.png', i, num_channels);

        return state_tensor.to(self.device)

    def DrawChannelImage(self, image_path, index, total_images):
        global win, win_size
        # Assume the display dimensions
        display_width, display_height = win_size

        # Calculate grid position based on index
        grid_cols = 3  # Number of columns in the grid
        grid_rows = 4  # Number of rows in the grid
        grid_width = display_width // grid_cols
        grid_height = display_height // grid_rows

        col = (index - 1) % grid_cols
        row = (index - 1) // grid_cols

        x = col * grid_width
        y = row * grid_height

        # Load the image
        try:
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, (grid_width, grid_height))  # Scale image to fit grid
        except pygame.error as e:
            print(f"Unable to load image: {image_path}. Error: {e}")
            return

        # Blit image at calculated position
        win.blit(image, (x, y))

        # If it's the last image in the current batch, update the display and wait for a key press
        if index % (grid_cols * grid_rows) == 0 or index == total_images:
            pygame.display.flip()

    def replaceThisCell(self, row, col, new_type):
        """
        Replace the cell at the specified row and column with the new type.
        """
        if 0 <= row < len(self.map_layout) and 0 <= col < len(self.map_layout[0]):
            self.map_layout[row] = self.map_layout[row][:col] + new_type + self.map_layout[row][col + 1:]

    def moveAllMobs(self, mob_type, allowed_tile):
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        mobs_positions = [(r, c) for r in range(rows) for c in range(cols) if self.map_layout[r][c] == mob_type]

        for r, c in mobs_positions:
            random.shuffle(directions)  # Shuffle directions to randomize mob movement
            moved = False
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols and self.map_layout[new_r][new_c] == allowed_tile:
                    # Move mob to new position
                    self.map_layout[r] = self.map_layout[r][:c] + allowed_tile + self.map_layout[r][c + 1:]
                    self.map_layout[new_r] = self.map_layout[new_r][:new_c] + mob_type + self.map_layout[new_r][
                                                                                         new_c + 1:]
                    moved = True
                    break  # Break after moving to avoid trying other directions

            if not moved:
                # If the mob cannot move (all adjacent tiles are not allowed), it stays in its current position
                continue

    def replaceAllCells(self, from_type, to_type):
        """
        Replace all instances of one cell type with another throughout the map.
        """
        for row in range(len(self.map_layout)):
            self.map_layout[row] = self.map_layout[row].replace(from_type, to_type)

    def GetDefaultPlayerPosition(self):
        """
        Finds the Start tile ('S') in the map layout and calculates the state for it.

        Returns:
            int: The state corresponding to the Start tile's position.
        """
        for row_index, row in enumerate(self.map_layout):
            if 'S' in row:
                col_index = row.index('S')
                return row_index * len(self.map_layout[0]) + col_index
        # Fallback in case 'S' is not found, though this should not happen
        return 0

    def reset(self):
        self.SetupLevel(self.level_index)
        self.render();
        state_tensor = self.generate_multi_channel_state()
        return self.state, state_tensor, {}
    def resetTestLevel(self):
        self.SetupTestLevel()
        self.render();
        state_tensor = self.generate_multi_channel_state()
        return self.state, state_tensor, {}

    def renderMap(self):
        def load_image(skin, name):
            """Loads an image from the 'tiles/' directory."""
            if(skin==""):
                return pygame.image.load(f'tiles/{name}.png')
            else:
                return pygame.image.load(f'tiles/{skin}/{name}.png')

        scale_factor = 2;

        def load_image_scaled(skin, image_name, scale_factor=2):
            # Load the image using your existing load_image function
            image = load_image(skin, image_name)
            # Get the current size of the image
            original_size = image.get_size()
            # Calculate the new size based on the scale factor
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            # Scale the image to the new size
            scaled_image = pygame.transform.scale(image, new_size)
            return scaled_image

        # Load tiles with added default and wall tiles
        tiles = {
            ',': load_image_scaled('', 'cursor', scale_factor),
            'CL': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_left', scale_factor),
            'CB': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_bottom', scale_factor),
            'CR': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_right', scale_factor),
            'CT': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_top', scale_factor),

            'S': load_image_scaled(self.skin, 'start', scale_factor),
            '.': load_image_scaled(self.skin, 'free', scale_factor),
            'H': load_image_scaled(self.skin, 'hole', scale_factor),
            'G': load_image_scaled(self.skin, 'goal', scale_factor),
            'C': load_image_scaled(self.skin, 'goalClosed', scale_factor),
            'K': load_image_scaled(self.skin, 'key', scale_factor),

            'J': load_image_scaled(self.skin, 'key2O', scale_factor),  # some level door - key
            'T': load_image_scaled(self.skin, 'key2L', scale_factor),
            # Chest, that contains some level door key, locked. must be unlocked with a button. After unlocking - transforms to a key
            'R': load_image_scaled(self.skin, 'press', scale_factor),
            # button to unlock a chest, that kontains a some level door
            'L': load_image_scaled(self.skin, 'door', scale_factor),  # some level door - opened
            'A': load_image_scaled(self.skin, 'doorClosed', scale_factor),  # some level door - closed
            'Q': load_image_scaled(self.skin, 'mob3', scale_factor),  # monster, that drops a key for some level door

            'M': load_image_scaled(self.skin, 'mob', scale_factor),
            'F': load_image_scaled(self.skin, 'mob2', scale_factor),
            'U': load_image_scaled(self.skin, 'unstable', scale_factor),
            'D': load_image_scaled(self.skin, 'dizzy', scale_factor),
            'P': load_image_scaled(self.skin, 'potion', scale_factor),
            'B': load_image_scaled(self.skin, 'bomb', scale_factor),
            'W': load_image_scaled(self.skin, 'wall', scale_factor),
            'WL': load_image_scaled(self.skin, 'wallL', scale_factor),  # Wall Top Left
            'WR': load_image_scaled(self.skin, 'wallR', scale_factor),  # Wall Top Right
        }

        state = self.state
        map_layout = self.map_layout
        grid_size_rows = len(self.map_layout)
        grid_size_cols = len(self.map_layout[0])

        current_row = self.state // grid_size_cols
        current_col = self.state % grid_size_cols

        # Calculate offset needed to center the character
        offset_grid_row = 0
        offset_grid_col = 0
        if (grid_size_rows > 9 or grid_size_cols > 9):
            offset_grid_row = - current_row
            offset_grid_col = - current_col

            # Max tiles on screen
            viewport_tiles_x = 11
            viewport_tiles_y = 15

            # Maximum allowable offsets given the level size - 4 and 2 - magic numbers
            max_offset_x = grid_size_cols - viewport_tiles_x // 2 - 4 + 4
            max_offset_y = grid_size_rows - viewport_tiles_y // 2 - 2 + 4

            # Calculate desired offset to center the character
            desired_offset_col = -current_col
            desired_offset_row = -current_row

            # Clamp the offsets to the maximum allowable values to avoid showing too much empty space
            offset_grid_col = max(desired_offset_col, -max_offset_x)
            offset_grid_row = max(desired_offset_row, -max_offset_y)

        # Assume each tile has fixed size for simplicity
        tile_width, tile_height = 32 * scale_factor, 16 * scale_factor

        # This time, calculate offset to center the (-1, -1) tile
        # First, find the center of the window
        center_x = win_size[0] / 2
        center_y = win_size[1] / 2

        # Calculate offset_x such that (-1, -1) tile's center is at window's center
        # We adjust for half a tile width because (-1, -1) is off-center to the left
        # and we move it up by half the total map height in isometric projection to align top
        offset_x = center_x - (tile_width // 2)

        # Calculate offset_y to place (-1, -1) at the top of the screen
        # We take into account the entire height of the map in isometric projection
        # and adjust it so the top is at the center_y, moving it up by half the height of one tile
        total_height_iso = (grid_size_rows * tile_height // 2) * 2  # Total height in isometric view
        offset_y = tile_height * 2 + tile_height * 3 + 160

        win.fill((0, 0, 0))  # Clear the screen

        def drawChar(row, col):

            tile_type = "CR"
            if (self.last_action == 0):
                tile_type = "CL"
            elif (self.last_action == 1):
                tile_type = "CB"
            elif (self.last_action == 2):
                tile_type = "CR"
            elif (self.last_action == 3):
                tile_type = "CT"
            this_tile_offset_y = tiles[tile_type].get_size()[1] + 16 * scale_factor
            this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
            # Convert grid coordinates to isometric, including walls
            iso_x = ((col + offset_grid_col) - (row + offset_grid_row)) * (
                    tile_width // 2) + offset_x + tile_width / 2 - this_tile_offset_x
            iso_y = ((col + offset_grid_col) + (row + offset_grid_row)) * (
                    tile_height // 2) + offset_y - this_tile_offset_y
            win.blit(tiles[tile_type], (iso_x, iso_y))

        def drawCursor(iso_x, iso_y):
            tile_type = ","
            this_tile_offset_y = tiles[tile_type].get_size()[1]
            this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
            win.blit(tiles[tile_type], (iso_x - this_tile_offset_x, iso_y - this_tile_offset_y))

        # Render tiles in isometric view, including boundary for walls
        for i in range(-1, grid_size_rows):
            for j in range(-1, grid_size_cols):
                tile_type = '.'
                # Determine the type of tile
                if i == -1 or j == -1:
                    if (j) == (-1):  # this must me in minus
                        tile_type = 'WL'  # Top left wall for the first cell
                    elif (i) == (-1):  # this must be in minus
                        tile_type = 'WR'  # Top right wall for the last cell in the first row

                else:
                    tile_type = map_layout[i][j]

                # Convert grid coordinates to isometric, including walls
                iso_x = ((j + offset_grid_col) - (i + offset_grid_row)) * (
                        tile_width // 2) + offset_x + tile_width / 2
                iso_y = ((j + offset_grid_col) + (i + offset_grid_row)) * (
                        tile_height // 2) + offset_y

                if (self.NeedToDrawEmptyTileUnder(tile_type)):
                    this_tile_offset_y = tiles['.'].get_size()[1]
                    this_tile_offset_x = tiles['.'].get_size()[0] / 2
                    win.blit(tiles['.'], (iso_x - this_tile_offset_x, iso_y - this_tile_offset_y))

                if (self.IsUnderTile(tile_type)):
                    this_tile_offset_y = tiles[tile_type].get_size()[1]
                    this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
                    win.blit(tiles[tile_type], (iso_x - this_tile_offset_x, iso_y - this_tile_offset_y))

                if (current_col == j and current_row == i):
                    drawChar(i, j);

                if (self.IsOverTile(tile_type)):
                    this_tile_offset_y = tiles[tile_type].get_size()[1]
                    this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
                    win.blit(tiles[tile_type], (iso_x - this_tile_offset_x, iso_y - this_tile_offset_y))

                if (self.cursor_grid_x == j and self.cursor_grid_y == i):
                    drawCursor(iso_x, iso_y);

            # Define your colors

        # Load the frame image and scale it to the window size
        frame_image = pygame.image.load('tiles/frame.png')
        # Draw (blit) the frame image over everything else
        win.blit(frame_image, (0, 0))

    def renderEditor(self, event, buttonDown):
        global win, win_size

        clock = pygame.time.Clock()

        self.draw_tile_selection_area();
        if(buttonDown):
            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                self.handle_mouse_click(x, y)
                self.handle_mouse_move(x, y)
        else:
            x, y = pygame.mouse.get_pos()
            self.handle_mouse_move(x, y)

        self.renderMap();

        descr = "Editor: Level " + str(self.level_index) + "\n";

        self.SetDescription(descr);

        self.draw_tile_selection_area();


        self.SetHeader();
        self.SetFooter();

        pygame.display.flip()
        clock.tick(10)  # Cap the frame rate

    def render(self):
        global win, win_size
        if self.enableRendering == False:
            return;

        clock = pygame.time.Clock()


        state = self.state
        map_layout = self.map_layout
        grid_size_rows = len(self.map_layout)
        grid_size_cols = len(self.map_layout[0])

        current_row = self.state // grid_size_cols
        current_col = self.state % grid_size_cols

        self.renderMap();

        descr = "Level " + str(self.level_index) + "  " + str(current_col) + "x" + str(current_row) + "\n";
        descr += "Steps " + str(self.step_count) + "\n";
        if (self.has_antidot):
            descr += "Has antidot!\n";
        if (self.is_dizzy):
            descr += "Poisoned!\n";
        if (self.key_count>0):
            descr += f"Keys: {self.key_count}\n";

        self.SetDescription(descr);

        self.SetHeader();
        self.SetFooter();

        pygame.display.flip()
        clock.tick(10)  # Cap the frame rate

    def draw_text_with_gradient(self, text, position, top_color, bottom_color, font_size=18):
        global win, win_size

        font_path = "font/solstice-nes.ttf"

        # Load the custom font
        font = pygame.font.Font(font_path, font_size)

        # Split the text into lines based on "\n"
        lines = text.split("\n")

        # Starting Y position for the first line
        y_pos = position[1]

        for line in lines:
            # Initial X position for the line
            x_pos = position[0]

            # Split the line into segments based on "=" sign
            segments = line.split("=")
            for i, segment in enumerate(segments):
                if i > 0:  # For segments following "=" signs, render the first character in white
                    first_char_surface = font.render(segment[0], True, pygame.Color('white'))
                    win.blit(first_char_surface, (x_pos, y_pos))
                    x_pos += first_char_surface.get_width()
                    segment = segment[1:]  # Remove the first character since it's already rendered

                # Render the remaining segment with gradient
                if segment:  # Check if segment is not empty
                    text_surface = font.render(segment, True, pygame.Color('white'))
                    gradient_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
                    for y in range(text_surface.get_height()):
                        # Calculate the color for the current position
                        alpha = y / text_surface.get_height()
                        color = [top_color[j] * (1 - alpha) + bottom_color[j] * alpha for j in range(3)]
                        pygame.draw.line(gradient_surface, color, (0, y), (text_surface.get_width(), y))
                    gradient_surface.blit(text_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                    win.blit(gradient_surface, (x_pos, y_pos))
                    x_pos += gradient_surface.get_width()

            # Move Y position down for the next line
            y_pos += text_surface.get_height()

    def RenderScreen(self, descr, avatar):
        global win, win_size

        # Load the frame image and scale it to the window size
        frame_image = pygame.image.load('tiles/frame.png')
        # Draw (blit) the frame image over everything else
        win.blit(frame_image, (0, 0))

        # Load the frame image and scale it to the window size
        avatar_image = pygame.image.load('tiles/' + avatar + '.jpg')
        # Define the target rectangle for the avatar
        x, y, x2, y2 = 591, 590, 722, 689
        target_width = x2 - x
        target_height = y2 - y

        # Calculate the scale factor to fit the avatar into the target rectangle
        # Preserve the aspect ratio of the avatar
        avatar_width = avatar_image.get_width()
        avatar_height = avatar_image.get_height()
        scale_factor = min(target_width / avatar_width, target_height / avatar_height)

        # Scale the avatar image
        scaled_avatar = pygame.transform.scale(avatar_image,
                                               (int(avatar_width * scale_factor), int(avatar_height * scale_factor)))

        # Calculate the top-left position to center the avatar in the target rectangle
        # This is optional if you want the avatar to be centered within the target area
        avatar_x = x + (target_width - scaled_avatar.get_width()) // 2
        avatar_y = y + (target_height - scaled_avatar.get_height()) // 2

        # Draw (blit) the scaled avatar image over everything else at the calculated position
        win.blit(scaled_avatar, (avatar_x, avatar_y))

        self.SetDescription(descr);
        self.SetHeader();
        self.SetFooter();

        pygame.display.flip()

    def close(self):
        global pygame, win, win_size;
        pygame.quit()
        # Placeholder for any cleanup tasks.
        # For example, if using Pygame for rendering:
        # pygame.quit()
        pass

    def DisableDisplay(self):
        self.enableRendering = False

    def EnableDisplay(self):
        self.enableRendering = True

    def SetDescription(self, param):
        self.draw_text_with_gradient(
            param, (30, 604),
            (193, 223, 254),
            (29, 99, 214))
        pass

    def SetTitle(self, param):
        print(param)
        pygame.display.set_caption("Solstice: " + param)
        pass

    def Won(self):
        print("Congratulations, you've reached the goal!")
        return self.NextLevel()

    def Lost(self):
        print("Oops, you fell into a hole or died!")
        return self.reset()

    def NextLevel(self):
        self.level_index = self.level_index + 1;
        return self.reset()

    def PrevLevel(self):
        self.level_index = self.level_index - 1;
        return self.reset()

    def SetHeader(self):

        # Level header
        self.draw_text_with_gradient(
            "" + str(self.level_name), (30, 30),
            (231, 255, 165),
            (0, 150, 0))

    def SetFooter(self):

        self.draw_text_with_gradient(
            "=Train e=Xpert =Eval =Reset =Next =Prev =Chan =Music E=dit", (8, 714),
            (231, 255, 165),
            (0, 150, 0), 14)

    def SetupLevel(self, level_index):
        self.level_index = level_index;
        self.map_layout = self.load_map_layout(self.level_index)
        self.used_channel_indices = self.GetUsedChannelIndexes()
        self.level_channels_count = len(
            self.used_channel_indices) + 1  # adding plus one for the player position on map.

        self.level_size_for_hidden_layer = len(self.map_layout) * len(self.map_layout[0]) * self.level_channels_count;
        self.level_height = len(self.map_layout)
        self.level_width = len(self.map_layout[0])

        self.state = self.GetDefaultPlayerPosition()
        self.done = False
        self.step_count = 0
        self.is_dizzy = False
        self.has_antidot = False
        self.key_count = 0

    def SetupTestLevel(self):
        self.used_channel_indices = self.GetUsedChannelIndexes()
        self.level_channels_count = len(
            self.used_channel_indices) + 1  # adding plus one for the player position on map.

        self.level_size_for_hidden_layer = len(self.map_layout) * len(self.map_layout[0]) * self.level_channels_count;
        self.level_height = len(self.map_layout)
        self.level_width = len(self.map_layout[0])

        self.state = self.GetDefaultPlayerPosition()
        self.done = False
        self.step_count = 0
        self.is_dizzy = False
        self.has_antidot = False
        self.key_count = 0

    def GetUsedChannelIndexes(self):
        valuable_channel_indices = {
            # 'S': 0,  # Start
            # '.': 1,  # Free space
            'H': 2,  # Hole
            'G': 3,  # Goal
            'K': 4,  # Key
            'C': 5,  # Closed goal
            # 'M': 6,  # Monster
            'F': 7,  # Monster with key drop
            'U': 8,  # Unstable
            # 'D': 9,  # Dizzy
            # 'P': 10,  # Potion
            'B': 11,  # Bomb
            'W': 12,  # Wall

            'J': 13,  # Door keys
            'T': 14,  # Chest with key
            'R': 15,  # Chest opener
            'L': 16,  # Opened door
            'A': 17,  # Closed door
            'Q': 18,  # Monster -door key
        }

        # Initialize a set to hold the unique tiles found in the map layout
        used_tiles = set()

        # Iterate through each row and column in the map layout
        for row in self.map_layout:
            for cell in row:
                if cell in valuable_channel_indices:
                    # If the cell/tile is in our list of valuable tiles, add it to the set
                    used_tiles.add(cell)

        # Ensure certain conditions are met
        if 'B' in used_tiles and 'K' not in used_tiles:
            used_tiles.add('K')  # Add 'K' if 'B' is present
        if 'C' in used_tiles and 'G' not in used_tiles:
            used_tiles.add('G')  # Add 'G' if 'C' is present
        if 'U' in used_tiles and 'H' not in used_tiles:
            used_tiles.add('H')  # Add 'H' if 'U' is present
        if 'T' in used_tiles and 'J' not in used_tiles:
            used_tiles.add('J')
        if 'A' in used_tiles and 'L' not in used_tiles:
            used_tiles.add('L')
        if 'Q' in used_tiles and 'J' not in used_tiles:
            used_tiles.add('J')


        # Ensure free space ('.') is always added
        # used_tiles.add('.')

        # Filter the valuable_channel_indices to include only those used in the map layout
        used_channel_indices = {key: valuable_channel_indices[key] for key in used_tiles}

        # Normalize the channel indices to be continuous starting from 0
        normalized_channel_indices = {key: i for i, key in enumerate(used_channel_indices)}

        # Sort the tiles based on some criterion, here alphabetically
        sorted_tiles = sorted(normalized_channel_indices)

        # Assign sorted indices
        sorted_channel_indices = {tile: index for index, tile in enumerate(sorted_tiles)}
        return sorted_channel_indices

