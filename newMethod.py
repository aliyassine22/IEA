import numpy as np
import streamlit as st
import time

# Constants
GRID_SIZE = 12
MOVE_COST = 1
ROBOT_COUNT = 3
ROUNDS = 10
TERMINALS = [(10, 10, 100), (2, 2, -50)]
BLOCKS = [(5, 5), (6, 6), (7, 7)]

def create_grid(grid_size, terminals, blocks):
    grid = np.zeros((grid_size, grid_size))
    policy_grid = np.full((grid_size, grid_size), ' ')
    
    for x, y, value in terminals:
        grid[x, y] = value
    
    for x, y in blocks:
        grid[x, y] = None  # Blocks are obstacles
    
    return grid, policy_grid

def get_neighbors(x, y, grid_size):
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    return [(x + dx, y + dy) for dx, dy in moves if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size]

def value_iteration(grid, move_cost, rounds):
    values = np.copy(grid)
    policy = np.full(grid.shape, ' ')
    
    for _ in range(rounds):
        new_values = np.copy(values)
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] is None or (x, y) in TERMINALS:
                    continue  # Skip blocks and terminal states
                
                neighbors = get_neighbors(x, y, grid.shape[0])
                best_value, best_move = float('-inf'), None
                
                for nx, ny in neighbors:
                    reward = values[nx, ny] - move_cost
                    if reward > best_value:
                        best_value, best_move = reward, (nx, ny)
                
                new_values[x, y] = best_value
                
                # Store movement direction instead of tuple
                direction_map = {
                    (0, 1): '↓', (0, -1): '↑', 
                    (1, 0): '→', (-1, 0): '←'
                }
                if best_move:
                    policy[x, y] = direction_map.get((best_move[0] - x, best_move[1] - y), ' ')
        
        values = new_values
    
    return values, policy

def move_robots(robots, policy):
    new_positions = []
    for x, y in robots:
        if policy[x, y] != ' ':
            new_positions.append((x + (1 if policy[x, y] == '→' else -1 if policy[x, y] == '←' else 0),
                                  y + (1 if policy[x, y] == '↓' else -1 if policy[x, y] == '↑' else 0)))
        else:
            new_positions.append((x, y))
    return new_positions

# Streamlit UI
st.title("Robot Pathfinding Simulation")

grid, policy_grid = create_grid(GRID_SIZE, TERMINALS, BLOCKS)
values, policy = value_iteration(grid, MOVE_COST, ROUNDS)

robots = [(0, 0), (3, 3), (8, 8)]

for _ in range(20):  # Simulate 20 steps
    st.write("## Grid State")
    grid_display = np.full((GRID_SIZE, GRID_SIZE), '⬜')
    for x, y in BLOCKS:
        grid_display[x, y] = '⬛'
    for x, y, _ in TERMINALS:
        grid_display[x, y] = '🏁'
    for x, y in robots:
        grid_display[x, y] = '🤖'
    st.write(grid_display)
    
    robots = move_robots(robots, policy)
    time.sleep(0.5)