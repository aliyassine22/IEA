import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import streamlit as st

class Agent:
    def __init__(self, x, y, agent_id):
        self.x = x
        self.y = y
        self.agent_id = agent_id
        self.previous_position=(x,y)
        self.path=[]

    def step():
        # compute forward move cost
        # compute left move cost
        # compute right move cost
        # compute stay in place cost
        print('test')
        # determine the maximum move cost    
    def __hash__(self): # allows agent objects to be used as dictionary keys also enables efficient comparison in sets
        return hash(self.agent_id)

    def __eq__(self, other): # to check if the two agents are the equal
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
    def __init__(self,shape):
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
    # print('ali check this')
    # for i in sorted_blocks:
    #     print(f"this is {i} " )
    lower_half = set(sorted_blocks[10:])
    print('lower half',lower_half)
    print('upper half', upper_half)
    print('shape positions: ',shape_positions)
    return upper_half, lower_half, shape_positions
    ##


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
    sorted_blocks = sorted(cross_coordinates, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
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
        #rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, color='red', alpha=0.6)     The following is used for the new ALgo
        rect = plt.Rectangle((x, y), 1, 1, color='red', alpha=0.6)
        ax.add_patch(rect)


# Plot agents as green circles
def plot_agents(ax, agent_positions, GRID_SIZE=10):

    for x, y in agent_positions:
        # circle = plt.Circle((x, y), 0.3, color='green', alpha=0.8)    The following is used for the new ALgo
        circle = plt.Circle((x+0.5, y+0.5), 0.3, color='green', alpha=0.8)
        ax.add_patch(circle)

# Plot weights as numbers over the grid
def plot_weights(ax, weight_map, GRID_SIZE=10):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            w = weight_map[(x, y)]
            ax.text(x+0.5, y+0.5, str(w), ha='center', va='center', fontsize=8, color='black')

# Combined function to plot the entire state on a single Axes
def plotState(ax, environment, agentList, GRID_SIZE=10):
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.clear() # clear the Axes to avoid overlapping drawings
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    agent_positions = [(agent.x, agent.y) for agent in agentList]
    weight_map = environment.weights
    # Draw all layers on the same Axes
    plot_shape(ax, environment.shape_positions, GRID_SIZE)   # colored destination cells
    plot_agents(ax, agent_positions, GRID_SIZE)  # agent circles
    plot_weights(ax, weight_map, GRID_SIZE)        # weight numbers overlay
    
    ax.set_aspect('equal', adjustable='box')
    return fig
    # st.pyplot(fig)
# Example usage with 20 agents

def run_simulation():
    import time

    st.title("Single Grid Simulation")
    shape_type = st.selectbox("Select the Shape", ["diamond", "rectangle", "cross"], index=0)

    # Initialize session state variables (update if shape changes)
    if 'grid' not in st.session_state or st.session_state.selected_shape != shape_type:
        st.session_state.selected_shape = shape_type  # store selected shape
        st.session_state.grid = Environment(shape=shape_type)
        st.session_state.agents = [Agent(x, 1-y, x + y) for y in range(2) for x in range(10)]
        st.session_state.step = 0
        st.session_state.running = False
        st.session_state.iter = 0  # persistent iteration counter

    AgentList = st.session_state.agents
    GRID = st.session_state.grid

    # Adjust the number of agents if necessary
    if len(GRID.shape_positions) < len(AgentList):
        AgentList = AgentList[:len(GRID.shape_positions)]

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_placeholder = st.empty()
    iter_placeholder = st.empty()

    # Display the current iteration (initial state should show 0)
    iter_placeholder.write(f"Step: {st.session_state.iter}")

    # Button to display the initial state (do not count this as a move)
    if st.button("Show Grid"):
        st.session_state.iter = 0  # Reset iteration counter to 0
        fig = plotState(ax, GRID, AgentList, GRID.GRID_SIZE)
        plot_placeholder.pyplot(fig)
        iter_placeholder.write(f"Step: {st.session_state.iter}")

    # Button to run the entire simulation automatically
    if st.button("Run Simulation"):
        st.session_state.running = True
        
        # Run simulation continuously
        while st.session_state.running:
            # Move agents first, then increment counter (so the initial state isn't counted)
            for agent in AgentList:
                takeMoveMoore(agent, GRID, AgentList)
            st.session_state.iter += 1  # Increment after agents have moved
            iter_placeholder.write(f"Step: {st.session_state.iter}")
            
            fig = plotState(ax, GRID, AgentList, GRID.GRID_SIZE)
            plot_placeholder.pyplot(fig)
            time.sleep(1)

        # After stopping, display the final state
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")
        fig = plotState(ax, GRID, AgentList, GRID.GRID_SIZE)
        plot_placeholder.pyplot(fig)

    # Button to run each step manually
    if st.button("Next Iteration"):
        # Move agents first, then increment counter
        for agent in AgentList:
            takeMoveMoore(agent, GRID, AgentList)
        st.session_state.iter += 1  # Increment the counter after moving
        iter_placeholder.write(f"Step: {st.session_state.iter}")
        fig = plotState(ax, GRID, AgentList, GRID.GRID_SIZE)
        plot_placeholder.pyplot(fig)

    # Button to stop the simulation
    if st.button("Stop Simulation"):
        st.session_state.running = False
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")
        fig = plotState(ax, GRID, AgentList, GRID.GRID_SIZE)
        plot_placeholder.pyplot(fig)

if __name__ == "__main__":
    run_simulation()


# if you get to a point where its not filling up use distance to remaining blocks to take the steps


#### FIRST FUNCTION IS TAKMEMOVECOST

### SECOND ONE IS BOOKING USING HUNGARIAN ALGORITHM AND TAKEMOVEA*  DONE


#### THIRD ONE IS 


import numpy as np
from scipy.optimize import linear_sum_assignment
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



## trying the new association
# def hungarianAssignment(agentList, shape_positions):   
#     """
#     Assigns each agent to a destination using the Hungarian algorithm based on Manhattan distance.

#     Parameters:
#         agentList (list): List of Agent objects (each with attributes x and y).
#         shape_positions (set): Set of (x,y) tuples representing destinations.

#     Returns:
#         dict: A mapping from each agent to its assigned destination.
#     """
#     # Convert shape positions to a sorted list for consistency
#     shape_positions_list = sorted(list(shape_positions), key=lambda pos: pos[1])

#     # Split destinations into upper and lower halves
#     upperDestination = shape_positions_list[:10]  # First 10 elements
#     lowerDestination = shape_positions_list[10:]  # Last 10 elements

#     # Split agents into upper and lower groups
#     upperAgents = [agent for agent in agentList if agent.y == 0]  # Agents at y=1
#     lowerAgents = [agent for agent in agentList if agent.y == 1]  # Agents at y=0

#     # Ensure there are 10 agents and 10 destinations in each group
#     if len(upperAgents) != 10 or len(lowerAgents) != 10:
#         raise ValueError("Mismatch between agents and destinations!")

#     # Create two separate 10x10 cost matrices
#     upper_cost_matrix = np.zeros((10, 10))
#     lower_cost_matrix = np.zeros((10, 10))

#     # Compute Manhattan distances for upper agents
#     for i, pos in enumerate(upperDestination):
#         for j, agent in enumerate(upperAgents):
#             upper_cost_matrix[i, j] = abs(agent.x - pos[0]) + abs(agent.y - pos[1])

#     # Compute Manhattan distances for lower agents
#     for i, pos in enumerate(lowerDestination):
#         for j, agent in enumerate(lowerAgents):
#             lower_cost_matrix[i, j] = abs(agent.x - pos[0]) + abs(agent.y - pos[1])

#     # Solve assignment using Hungarian algorithm
#     row_ind_upper, col_ind_upper = linear_sum_assignment(upper_cost_matrix)
#     row_ind_lower, col_ind_lower = linear_sum_assignment(lower_cost_matrix)

#     # Map agents to destinations
#     Dict_Agent_Destination = {}
#     for i, j in zip(row_ind_upper, col_ind_upper):
#         Dict_Agent_Destination[upperAgents[j]] = upperDestination[i]

#     for i, j in zip(row_ind_lower, col_ind_lower):
#         Dict_Agent_Destination[lowerAgents[j]] = lowerDestination[i]

#     return Dict_Agent_Destination
