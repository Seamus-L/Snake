#Import for the python game
import pygame
import random


#Import for ML
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#####################
#Define ML stuff

class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)  # Output layer
        return x

# Set parameters
input_size = 162  # Number of inputs - direction of travel, and grid of game state
hidden_size = 16  # Number of neurons in the hidden layer
output_size = 4  # Number of outputs (up, down, left, right)

# Initialize the network
net = SnakeNet()

# Define optimizer and loss function
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()  # For a reinforcement learning scenario, you might use a custom loss

















#####################
#Define game

# Initialize Pygame
pygame.init()

# Set up display
cell_size = 50  # Size of each cell in pixels
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Set up clock
clock = pygame.time.Clock()
fps = 10  # Frames per second

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LGREY = (192, 192, 192)
DGREY = (64, 64, 64)
RED = (255, 0, 0)

# Define cell grid size
cols = width // cell_size
rows = (height - 100) // cell_size  # Adjust rows for the top black bar

# Define possible cells for use later
possible_cells = [(col, row) for col in range(cols) for row in range(rows)]

# Initial position for the player (in cells)
player_cell = [3, 5]
player_color = (100, 255, 100)

# Initialize direction - 1: right, 2: down, 3: left, 4: up
direction = 1

# Dictionary relating movement to direction
movement = {
    1: (1, 0),   # Right
    2: (0, 1),   # Down
    3: (-1, 0),  # Left
    4: (0, -1)   # Up
}

# Font setup
font = pygame.font.Font(None, 36)  # Use default font, size 36

# Initialize score and time
score = 0
start_ticks = pygame.time.get_ticks()
high_score = 0

# Initialize the game to spawn in points for the player to grab
point = 1
point_cell = [10, 5]
circle_x = point_cell[0] * cell_size + cell_size // 2
circle_y = point_cell[1] * cell_size + 100 + cell_size // 2

# Define dead event
DEAD_EVENT = pygame.USEREVENT + 1

# Game loop flags
running = True
paused = False  # Flag for pause state
dead = False    # Flag for dead state

# Initialize array that holds tail
past_cells = [player_cell]
tail = 1  # Length of tail
tail_cells = [[2, 5]]






#For ML
grid_size = (cols, rows)
grid = torch.zeros(grid_size) 



def get_game_state(player_cell, tail_cells, point_cell, direction):
    grid = torch.zeros(grid_size) 

    # Get direction of travel
    snake_direction = torch.tensor(movement[direction], dtype=torch.float32)  # Convert direction to tensor

    # Get location of head
    grid[player_cell[0], player_cell[1]] = 1  # 1 represents head

    # Get locations of tail
    for (x, y) in tail_cells:
        grid[x, y] = 2  # 2 represents the snake's body

    # Get location of food
    grid[point_cell[0], point_cell[1]] = 3  # 3 represents food

    # Flatten the grid to a 1D tensor
    grid_flattened = grid.view(-1)  # Flatten the grid to 1D tensor

    # Concatenate the flattened grid and direction tensor
    state = torch.cat([grid_flattened, snake_direction])

    return state



def choose_action(output):
    # Ensure output is 2D for torch.max, or use dim=0 for 1D tensors
    if output.dim() == 1:
        _, predicted_direction = torch.max(output, dim=0)  # Use dim=0 for 1D tensors
    else:
        _, predicted_direction = torch.max(output, dim=1)  # Use dim=1 for 2D tensors

    action = predicted_direction.item()  # Convert tensor to Python int
    return action








# Game loop
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle pause state with spacebar
                paused = not paused
            elif event.key == pygame.K_q and paused:  # Quit game from pause state
                running = False
        elif event.type == DEAD_EVENT:
            print("Custom DEAD event triggered!")
            dead = True  # Set a flag or handle the event as needed

    if paused:
        # Draw pause message on top of the current screen content
        pause_text1 = font.render("The game is paused.", True, WHITE)
        pause_text2 = font.render("Press space to unpause", True, WHITE)
        pygame.draw.rect(screen, (128, 128, 128), (width // 3 - 10, height // 2 - 70, 290, 100))
        screen.blit(pause_text1, (width // 3, height // 2 - 50))
        screen.blit(pause_text2, (width // 3, height // 2))
    else:
        if not dead:
            # For ML
            state = get_game_state(player_cell, tail_cells, point_cell, direction)
            state = state.unsqueeze(0)  # Add batch dimension if required
            output = net(state)
            print(output.shape)  # Print shape to verify


            # Choose an action based on the network's output
            action = choose_action(output)

            print(action)

            # Update direction based on action
            if action == 0 and direction != 3:
                direction = 1  # Right
            elif action == 1 and direction != 4:
                direction = 2  # Down
            elif action == 2 and direction != 1:
                direction = 3  # Left
            elif action == 3 and direction != 2:
                direction = 4  # Up
            # Note: Adjust action indices if needed




            # Handle keys
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT] and direction != 3:
                direction = 1
            if keys[pygame.K_DOWN] and direction != 4:
                direction = 2
            if keys[pygame.K_LEFT] and direction != 1:
                direction = 3
            if keys[pygame.K_UP] and direction != 2:
                direction = 4

            # Update previous coordinates
            if len(past_cells) < 160:
                past_cells.insert(0, player_cell)
            else:
                past_cells.insert(0, player_cell)
                past_cells.pop()

            if len(tail_cells) != tail:
                tail_cells.append([])

            for i in range(tail):
                tail_cells[i] = past_cells[i]

            # Update player cell position
            new_player_cell = [
                player_cell[0] + movement[direction][0],
                player_cell[1] + movement[direction][1]
            ]

            # Check boundaries and handle dead condition after movement
            if (new_player_cell[0] < 0 or new_player_cell[0] >= cols or
                new_player_cell[1] < 0 or new_player_cell[1] >= rows):
                dead = True
            else:
                player_cell = new_player_cell

            # Check to see if player is on tail
            if new_player_cell in tail_cells:
                dead = True

            # Convert cell position to pixel position
            player_x = player_cell[0] * cell_size
            player_y = player_cell[1] * cell_size + 100  # Offset by 100 pixels for the bar

            # Draw the tiled background
            for row in range(rows):
                for col in range(cols):
                    # Determine color based on row and column
                    if (row + col) % 2 == 0:
                        color = LGREY
                    else:
                        color = DGREY

                    # Draw the background cell
                    pygame.draw.rect(screen, color, (col * cell_size, row * cell_size + 100, cell_size, cell_size))

            # Draw the black bar at the top
            pygame.draw.rect(screen, BLACK, (0, 0, width, 100))

            # Draw the player's square on top of the background
            pygame.draw.rect(screen, player_color, (player_x, player_y, cell_size, cell_size))

            # Draw the tail squares on top of the background
            for i in range(tail):
                if i % 2 != 0:
                    tail_colour = (100, 240, 100)
                else:
                    tail_colour = (80, 200, 80)
                pygame.draw.rect(screen, tail_colour, (tail_cells[i][0] * cell_size, tail_cells[i][1] * cell_size + 100, cell_size, cell_size))

            # Check to see if player earned a point
            if player_cell == point_cell:
                score += 1
                tail += 1  # Add element to tail
                point = 0

            # Check if there is a point on the playing field currently
            if point == 0:
                # Generate list of open cells
                open_cells = list(set(possible_cells) - {tuple(player_cell)} - set(map(tuple, tail_cells)))

                # Spawn a red circle on a random open cell
                point_cell = list(random.choice(open_cells))

                # Convert cell coordinates to pixel coordinates
                circle_x = point_cell[0] * cell_size + cell_size // 2
                circle_y = point_cell[1] * cell_size + 100 + cell_size // 2

                point = 1

            # Draw the red circle for the point
            pygame.draw.circle(screen, RED, (circle_x, circle_y), 20)

            # Calculate elapsed time using pygame.time.get_ticks()
            elapsed_ticks = pygame.time.get_ticks() - start_ticks
            elapsed_seconds = elapsed_ticks // 1000

            # Render SCORE and TIME text
            score_text = font.render(f"SCORE: {score}", True, WHITE)
            time_text = font.render(f"TIME: {elapsed_seconds}", True, WHITE)
            high_score_text = font.render(f"HIGH SCORE: {high_score}", True, WHITE)

            # Blit text onto the screen
            screen.blit(score_text, (250, 50))  # Position the score text
            screen.blit(time_text, (450, 50))  # Position the time text
            screen.blit(high_score_text, (600, 10))  # Position the high score text

            # Check to see if dead
            if dead:
                # Handle what should happen when DEAD event is triggered
                screen.fill((255, 0, 0))  # Example: Fill screen with red
                pygame.display.flip()  # Update display
                pygame.time.wait(1000)  # Wait for 1 second
                # Set highscore
                if score > high_score:
                    high_score = score
                # Reinitialize
                score = 0
                start_ticks = pygame.time.get_ticks()
                point_cell = [10, 5]
                circle_x = point_cell[0] * cell_size + cell_size // 2
                circle_y = point_cell[1] * cell_size + 100 + cell_size // 2
                player_cell = [3, 5]
                tail = 1
                tail_cells = []
                direction = 1
                dead = False  # Reset the flag
                pygame.display.flip()  # Update display

    # Update the display
    pygame.display.flip()

    # Tick the clock
    clock.tick(fps)

# Quit Pygame
pygame.quit()


