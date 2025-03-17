import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, x, y, agent_id):
        self.x = x
        self.y = y
        self.agent_id = agent_id
        self.previous_position=(x,y)
        self.path=[]
  
    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other):
        return isinstance(other, Agent) and self.agent_id == other.agent_id

def takeMoveVonNeumann(agent, env,agentList):
    occupiedSquares={(agent.x,agent.y)}
    for agentt in agentList:
        occupiedSquares.add((agentt.x,agentt.y))
    print(occupiedSquares)
    x_coordinate = agent.x
    y_coordinate = agent.y
    # possible moves are staying in place, increasing or decreasing x or increasing or decreasing y
    possible_new_positions={(x_coordinate,y_coordinate-1),(x_coordinate,y_coordinate+1),(x_coordinate-1,y_coordinate),(x_coordinate+1,y_coordinate)}
    # Remove invalid moves
    possible_new_positions = {
        (x, y) for (x, y) in possible_new_positions 
        if 0 <= x < env.GRID_SIZE and 0 <= y < env.GRID_SIZE
    }
    possible_new_positions -= occupiedSquares  # Remove occupied positions

    possible_new_positions.remove(agent.previous_position) # makes sure the agent doesnt go back to their previous position getting stuck in the loop
    print(possible_new_positions)
    # compute move_weights
    best_move=(x_coordinate,y_coordinate)
    move_reward=env.weights.get(best_move)
    print('my rewarddddd', move_reward)
    for tuple in possible_new_positions:
        new_move_reward= env.weights.get(tuple) 
        if new_move_reward> move_reward:
            best_move=tuple
            move_reward=new_move_reward
    print(f'Before: ({agent.x}, {agent.y}) -> After: ({best_move[0]}, {best_move[1]})')
    agent.previous_position=(agent.x,agent.y)
    print('k')
    print(agent.previous_position)
    agent.x,agent.y=best_move




     # try the CBS algorithm
     # try to implement the A* algorithm
    # if the move isnt better dont take it


def takeMoveMoore(agent, env,agentList):
    occupiedSquares={(agent.x,agent.y)}
    for agentt in agentList:
        occupiedSquares.add((agentt.x,agentt.y))
    
    # we will use this to generate if two distances are the same which path it should take
    unfilledDestinations=env.shape_positions-occupiedSquares
    print('unfilled',unfilledDestinations)


    print(occupiedSquares)
    x_coordinate = agent.x
    y_coordinate = agent.y
    # possible moves are staying in place, increasing or decreasing x or increasing or decreasing y 
    possible_new_positions={(x_coordinate,y_coordinate-1),(x_coordinate-1,y_coordinate),(x_coordinate+1,y_coordinate),(x_coordinate,y_coordinate+1),
                            (x_coordinate-1,y_coordinate-1),(x_coordinate+1,y_coordinate+1),(x_coordinate-1,y_coordinate+1),(x_coordinate+1,y_coordinate-1)}
    # Remove invalid moves
    possible_new_positions = {
        (x, y) for (x, y) in possible_new_positions 
        if 0 <= x < env.GRID_SIZE and 0 <= y < env.GRID_SIZE
    }
    possible_new_positions -= occupiedSquares  # Remove occupied positions

    if agent.previous_position in possible_new_positions and len(possible_new_positions) > 1:
        possible_new_positions.remove(agent.previous_position)
    # possible_new_positions.add((x_coordinate,y_coordinate)) # add a position to stay in place
    print(possible_new_positions)
    # compute move_weights
    best_move=(x_coordinate,y_coordinate)
    move_reward=env.weights.get(best_move)
    print('my rewarddddd', move_reward)
    for tuple in possible_new_positions:
        new_move_reward= env.weights.get(tuple) 
        if new_move_reward> move_reward:
            best_move=tuple
            move_reward=new_move_reward
        if new_move_reward==move_reward:
            #check which tuple reduces the distance to the closest unfilled destination
            for(ux,uy) in unfilledDestinations:
                if abs(ux-tuple[0])+abs(uy-tuple[1])<abs(ux-best_move[0])+abs(uy-best_move[1]):
                    best_move=tuple
                    move_reward=new_move_reward
    print(f'Before: ({agent.x}, {agent.y}) -> After: ({best_move[0]}, {best_move[1]})')
    agent.previous_position=(agent.x,agent.y)
    print(agent.previous_position)
    agent.x,agent.y=best_move
     

class Environment:
    def __init__(self,shape='diamond'):
        self.GRID_SIZE=10
        if shape=='diamond':
            upper_half, lower_half, shape_positions = generate_diamond(self.GRID_SIZE)
        if shape=='rectangle':
            shape_positions=generate_rectangle(self.GRID_SIZE)
            print('RECTANGLEEEEE BROOOOOOOOOO')
        elif shape=='cross':
            shape_positions=generate_cross(self.GRID_SIZE)
        
        self.weights=compute_weights(shape_positions,self.GRID_SIZE)
        self.shape_positions = shape_positions   
        
def generate_diamond(GRID_SIZE=10):
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    shape_positions = set()
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (-1, 2), (1, -2), (1, 2)
    ]
    for dx, dy in offsets:
        xx, yy = center_x + dx, center_y + dy
        if 0 <= xx < GRID_SIZE and 0 <= yy < GRID_SIZE:
            shape_positions.add((xx, yy))
    
    sorted_blocks = sorted(shape_positions, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, shape_positions


def generate_rectangle(GRID_SIZE=10):
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    rect_x = center_x - 2
    rect_y = center_y - 2
    rect_width = 4
    rect_height = 5

    rectangle_coordinates = set()
    for i in range(rect_width):
        for j in range(rect_height):
            rectangle_coordinates.add((rect_x + i, rect_y + j))

    return rectangle_coordinates


from collections import deque

def determineSquareCost(shapeCoordinates, grid_size=10):
    """
    Given a set of coordinates (x,y) that define a continuous shape within a 10x10 grid,
    assign each square in the shape a "depth-based" weight, with boundary squares having
    the smallest weights and squares deeper in the shape having larger weights.

    :param shapeCoordinates: set of (x, y) tuples that define the shape
    :param grid_size: default is 10 for a 10x10 grid
    :return: A dictionary mapping (x, y) -> weight
    """

    # Convert shapeCoordinates to a set for quick lookups
    shape = set(shapeCoordinates)

    # Identify boundary squares: squares in the shape that have at least
    # one neighbor outside the shape (or out of grid bounds).
    # For adjacency, we consider 4 directions (N, S, E, W).
    def neighbors(x, y):
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            yield nx, ny

    boundary_squares = []
    for (x, y) in shape:
        for nx, ny in neighbors(x, y):
            if (nx, ny) not in shape or not (0 <= nx < grid_size and 0 <= ny < grid_size):
                boundary_squares.append((x, y))
                break

    # Use multi-source BFS from all boundary squares to find the "depth" of each cell
    # in the shape. We'll store the depth in a dictionary called `depth_map`.
    depth_map = {}
    queue = deque()

    # Initialize the queue with all boundary squares at depth 0
    for bx, by in boundary_squares:
        depth_map[(bx, by)] = 0
        queue.append((bx, by))

    # BFS
    while queue:
        cx, cy = queue.popleft()
        current_depth = depth_map[(cx, cy)]
        for nx, ny in neighbors(cx, cy):
            # Only traverse neighbors that are inside the shape and not visited yet
            if (nx, ny) in shape and (nx, ny) not in depth_map:
                depth_map[(nx, ny)] = current_depth + 1
                queue.append((nx, ny))

    # Now each shape cell has a "depth" (distance from boundary).
    # We can convert this depth into a weight. For example, weight = depth + 1
    # so boundary squares get weight=1, next deeper squares get weight=2, etc.
    weight_map = {}
    for (x, y), d in depth_map.items():
        weight_map[(x, y)] = 100+d 

    print(weight_map)

    return weight_map




from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Provided helper to generate the diamond shape
def generate_diamond(GRID_SIZE=10):
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    shape_positions = set()
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (-1, 2), (1, -2), (1, 2)
    ]
    for dx, dy in offsets:
        xx, yy = center_x + dx, center_y + dy
        if 0 <= xx < GRID_SIZE and 0 <= yy < GRID_SIZE:
            shape_positions.add((xx, yy))
    
    sorted_blocks = sorted(shape_positions, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, shape_positions

# Compute the weight for each grid cell


def generate_cross(GRID_SIZE=10):
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    cross_coordinates = set()
    
    # 1) Vertical part: 2 wide (j in [-1..0]) and 6 high (i in [-3..2])
    for i in range(-3, 3):   # -3, -2, -1, 0, 1, 2  => 6 rows
        for j in range(-1, 1):  # -1, 0 => 2 columns
            cross_coordinates.add((center_x + j, center_y + i))
    
    # 2) Horizontal part: 1 wide (just the center row) and 6 columns
    #    i in [-3..2] => 6 cells
    for i in range(-3, 3):  # -3, -2, -1, 0, 1, 2 => 6 columns
        cross_coordinates.add((center_x + i, center_y))
    
    return cross_coordinates


def compute_weights(shape_positions, GRID_SIZE=10):
    """
    Computes weights for every cell in the grid.
      - For cells inside the diamond, the weight is higher the closer the cell is to the top (i.e. smallest y).
      - For cells outside the diamond, weight is based on distance (via BFS) to the nearest diamond cell.
    """

    # Determine min and max y in diamond cells.
    min_y_in_diamond = min(y for (x, y) in shape_positions)
    max_y_in_diamond = max(y for (x, y) in shape_positions)
    
    # BFS from all diamond cells to compute distance for every cell
    distance_map = {(x, y): None for x in range(GRID_SIZE) for y in range(GRID_SIZE)}
    queue = deque()
    for (dx, dy) in shape_positions:
        distance_map[(dx, dy)] = 0
        queue.append((dx, dy))
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        cx, cy = queue.popleft()
        for mx, my in directions:
            nx, ny = cx + mx, cy + my
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if distance_map[(nx, ny)] is None:
                    distance_map[(nx, ny)] = distance_map[(cx, cy)] + 1
                    queue.append((nx, ny))
    
    # Compute maximum distance for outside cells (for scaling)
    outside_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                     if (x, y) not in shape_positions]
    max_dist_outside = max(distance_map[(x, y)] for (x, y) in outside_cells) if outside_cells else 0

    weight_map = {}
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x, y) in shape_positions:
                # For diamond cells, higher weight for cells closer to the top (smaller y)
                depth_from_top = y - min_y_in_diamond
                weight_map[(x, y)] = 10 + (max_y_in_diamond + y)
            else:
                # For outside cells, closer cells to the diamond get higher weights.
                dist = distance_map[(x, y)]
                weight_map[(x, y)] = 1 + (max_dist_outside - dist)
        
    # weight_map[(5,5)]=21.5
    return weight_map

# Plot the diamond (colored squares)
def plot_shape(ax, shape_positions, GRID_SIZE=10):
    for x, y in shape_positions:
        # Draw filled square with some transparency
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='red', alpha=0.6)
        ax.add_patch(rect)

# Plot agents as green circles
def plot_agents(ax, agent_positions, GRID_SIZE=10):

    for x, y in agent_positions:
        circle = plt.Circle((x, y), 0.3, color='green', alpha=0.8)
        ax.add_patch(circle)

# Plot weights as numbers over the grid
def plot_weights(ax, weight_map, GRID_SIZE=10):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            w = weight_map[(x, y)]
            ax.text(x, y, str(w), ha='center', va='center', fontsize=8, color='black')

# Combined function to plot the entire state on a single Axes
def plotState(environment, agentList, GRID_SIZE=10):
    agent_positions = [(agent.x, agent.y) for agent in agentList]
    weight_map = environment.weights
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    
    # Draw all layers on the same Axes
    plot_shape(ax, environment.shape_positions, GRID_SIZE)   # colored destination cells
    plot_agents(ax, agent_positions, GRID_SIZE)  # agent circles
    plot_weights(ax, weight_map, GRID_SIZE)        # weight numbers overlay
    
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# Example usage with 20 agents

    
def run_simulation():
    
    import time

    AgentList = []
    i=0
    for y in range(2):  # First loop over rows
        for x in range(10):  # Then loop over columns
            AgentList.append(Agent(x, 1-y, i))
            i+=1

    GRID = Environment(shape='diamond')


    if(len(GRID.shape_positions)<len(AgentList)):
        AgentList=AgentList[0:len(GRID.shape_positions)]
    
    for i in range(20):
        plotState(GRID,AgentList)
        for agent in AgentList:
            # takeMoveVonNeumann(agent, GRID, AgentList)
            takeMoveMoore(agent, GRID, AgentList)
            print(i)
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()

# if you get to a point where its not filling up use distance to remaining blocks to take the steps


#### FIRST FUNCTION IS TAKMEMOVECOST

### SECOND ONE IS BOOKING USING HUNGARIAN ALGORITHM AND TAKEMOVEA*  DONE


#### THIRD ONE IS 


import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarianAssignment(agentList, shape_positions):   
    """
    Given a list of agents and a set of destination coordinates (shape_positions),
    assign each agent to a destination using the Hungarian algorithm based on Manhattan distance.

    Parameters:
        agentList (list): List of Agent objects (each must have attributes x and y).
        shape_positions (set): Set of (x,y) tuples representing destination positions.
    
    Returns:
        dict: A mapping from each agent to its assigned destination.
    """
    # Convert the shape positions to a list to have consistent ordering.
    shape_positions_list = list(shape_positions)
    
    # Number of destinations (rows) and number of agents (columns)
    rows = len(shape_positions_list)
    cols = len(agentList)
    
    # Create a cost matrix of size (rows x cols)
    # Use a list comprehension to avoid pitfalls with multiplying lists.
    cost_matrix = np.zeros((rows, cols))
    
    # Fill the cost matrix with Manhattan distances
    for i, pos in enumerate(shape_positions_list):
        for j, agent in enumerate(agentList):
            cost_matrix[i, j] = abs(agent.x - pos[0]) + abs(agent.y - pos[1])
    
    # Solve the assignment problem using the Hungarian algorithm.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build a dictionary mapping each agent  to its assigned destination.
    Dict_Agent_Destination = {}
    for i, j in zip(row_ind, col_ind):
        # shape_positions_list[i] is assigned to agentList[j]
        Dict_Agent_Destination[agentList[j]] = shape_positions_list[i]
    
    return Dict_Agent_Destination    ### AGENT TO COORDINATE DICTIONARY


