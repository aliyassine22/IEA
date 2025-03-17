import heapq
import time
from matplotlib import pyplot as plt
import numpy as np
from greedyWeightMapUI import Agent,plotState,Environment, plot_shape, plot_agents


####
class Node:
    def __init__(self, position, timestep, g, h, parent=None):
        self.position = position      # (x, y) tuple
        self.timestep = timestep      # integer time step
        self.g = g                    # cost from start to current node
        self.h = h                    # heuristic (estimated cost to goal)
        self.f = g + h                # total estimated cost
        self.parent = parent          # pointer to the parent node
        
    def __lt__(self, other):
        return self.f < other.f

def heuristic(position, goal):
    # Use Manhattan distance as heuristic
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def movement_cost(current, neighbor):
    # For grid movement with cardinal moves, cost is constant (1)
    return 1

def get_neighbors(position):
    x, y = position
    # Four cardinal directions; add diagonals if desired
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1),(x+1,y+1),(x-1,y-1),(x+1,y-1),(x-1,y+1)]
    
def is_valid(position, grid_size):
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size

def reconstruct_path(node):
    """
    Reconstructs the path from the start node to the given node.
    Returns a list of tuples: (position, timestep)
    """
    path = []
    while node is not None:
        path.append((node.position, node.timestep))
        node = node.parent
    path.reverse()
    return path

def plan_path(start_square, goal_square, start_timestep, occupied_set, grid_size):
    """
    Plans a path from start_square to goal_square using A* search while
    considering reserved positions (occupied_set) for future time steps.

    Parameters:
      start_square: (x, y) starting position
      goal_square: (x, y) goal position
      start_timestep: integer starting time step
      occupied_set: set of (position, timestep) tuples already reserved
      grid_size: size of the grid (assumed square grid: 0 to grid_size-1)

    Returns:
      A list of (position, timestep) tuples representing the planned path,
      or None if no path is found.
    """
    open_list = []
    closed_set = {}  # key: (position, timestep) -> Node
    
    start_node = Node(start_square, start_timestep, 0, heuristic(start_square, goal_square))
    heapq.heappush(open_list, start_node)
    closed_set[(start_square, start_timestep)] = start_node

    while open_list:
        current = heapq.heappop(open_list)
        
        # Check if we've reached the goal position
        if current.position == goal_square:
            return reconstruct_path(current)
        
        # Explore neighbors
        for neighbor in get_neighbors(current.position):
            next_timestep = current.timestep + 1
            
            if not is_valid(neighbor, grid_size):
                continue  # Skip invalid positions
            
            # Check if the position is reserved at next_timestep
            if (neighbor, next_timestep) in occupied_set:
                continue
            
            tentative_g = current.g + movement_cost(current.position, neighbor)
            neighbor_key = (neighbor, next_timestep)
            
            # If this neighbor state hasn't been visited or we found a cheaper path
            if neighbor_key not in closed_set or tentative_g < closed_set[neighbor_key].g:
                new_node = Node(neighbor, next_timestep, tentative_g,
                                heuristic(neighbor, goal_square), current)
                heapq.heappush(open_list, new_node)
                closed_set[neighbor_key] = new_node

    # If no path was found, return None
    return None

def reserve_path(path, occupied_set):
    """
    Reserves the path in the occupied_set.
    Each element in the path is a tuple (position, timestep).
    """
    for square, timestep in path:
        occupied_set.add((square, timestep))



# Combined function to plot the entire state on a single Axes
def plotState(environment, agentList,agent_assignment ,GRID_SIZE=10):
    agent_positions = [(agent.x, agent.y) for agent in agentList]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    
    # Draw all layers on the same Axes
    plot_shape(ax, environment.shape_positions, GRID_SIZE)   # colored destination cells
    plot_agents(ax, agent_positions, GRID_SIZE)  # agent circles
    plot_destinations(ax, agent_assignment, GRID_SIZE)        # weight numbers overlay
    plt.show()

def plot_destinations(ax,agent_assignments,GRID_SIZE):
      for agent, destination in agent_assignments.items():
        x_coor=agent.x
        y_coor=agent.y
        x_dest=destination[0]
        y_dest=destination[1]
        ax.text(x_coor, y_coor, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
        ax.text(x_dest, y_dest, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
        # ax.arrow(x_coor, y_coor, x_dest-x_coor, y_dest-y_coor, head_width=0.1, head_length=0.1, fc='black', ec='black')



############################################################################################################


def splitagents(agentList):
    upperList=[]
    lowerList=[]
    for agent in agentList:
        if agent.y==1:
            upperList.append(agent)
        else:
            lowerList.append(agent)
    
    print(len(lowerList)," ",len(upperList))
    return upperList,lowerList
def splitShape(shape_positions):
    UpperHalf=[]
    LowerHalf=[]
    for (x,y) in shape_positions:
        if y>=5:
            UpperHalf.append((x,y))
        else:
            LowerHalf.append((x,y))
    return UpperHalf,LowerHalf
def assignDestInLane(agentList, shape_positions):
    assignments = {}  # Avoid using 'dict' as a variable name
    upperList, lowerList = splitagents(agentList)
    
    # Create lane_map from shape_positions using x as the lane identifier.
    lane_map = {}
    for x, y in shape_positions:
        if x not in lane_map:
            lane_map[x] = []  # Initialize priority queue for the lane
        heapq.heappush(lane_map[x], -y)  # Use negative y for max-heap behavior
    print(lane_map)
    # Create an array to keep track of the number of agents in each lane
    arr = np.zeros(10, dtype=int)
    for x in lane_map:
        arr[x] = len(lane_map[x])
    print(arr) 
    # Process agents with matching lane exactly (using agent.x)
    for agent in upperList[:]:
        for i in range(len(arr)):
            if i == agent.x and i in lane_map and len(lane_map[i]) > 0:
                assignments[agent] = (i,-heapq.heappop(lane_map[i]))
                arr[i] -= 1
                upperList.remove(agent)
                break  # Stop once a match is found
    
    for agent in lowerList[:]:
        for i in range(len(arr)):
            if i == agent.x and i in lane_map and len(lane_map[i]) > 0:
                assignments[agent] = (i,-heapq.heappop(lane_map[i]))
                arr[i] -= 1
                lowerList.remove(agent)
                break  # Stop once a match is found
    for agent, destination in assignments.items():
        print(agent.x, agent.y, destination,'hh')
    # For remaining agents, find the closest available lane based on agent.x
    for agent in upperList:
        minDiff = float('inf')
        closest_x = -1
        for i in range(10):
            if i in lane_map and arr[i] > 0 and abs(agent.x - i) < minDiff:
                minDiff = abs(agent.x - i)
                closest_x = i
        if closest_x != -1:
            assignments[agent] = ( closest_x,-heapq.heappop(lane_map[closest_x]))
            arr[closest_x] -= 1
    
    for agent in lowerList:
        minDiff = float('inf')
        closest_x = -1
        for i in range(10):
            if i in lane_map and arr[i] > 0 and abs(agent.x - i) < minDiff:
                minDiff = abs(agent.x - i)
                closest_x = i
        if closest_x != -1:
            assignments[agent] = ( closest_x,-heapq.heappop(lane_map[closest_x]))
            arr[closest_x] -= 1
    
    return assignments



        
############################################################################################################


if __name__ == "__main__":
   
    AgentList = []
    i=0
    for y in range(2):  # First loop over rows
        for x in range(10):  # Then loop over columns
            AgentList.append(Agent(x, 1-y, i))
            i+=1

    GRID = Environment(shape='diamond')
    if(len(GRID.shape_positions)<len(AgentList)):
        AgentList=AgentList[0:len(GRID.shape_positions)]
    
    Agent_Assignemnt=assignDestInLane(AgentList,GRID.shape_positions)
    GRID_SIZE =GRID.GRID_SIZE
    reservation_table = set()
    
    
    # Define start and goal positions for an agent and the start time

    # sort the agent list based on two y_coordinate then closeness to the center
    for agent in AgentList:
        start_square = (agent.x, agent.y)
        goal_square = Agent_Assignemnt[agent]
        start_timestep = 0
        
        # Plan a path using A* search that avoids already reserved positions
        path = plan_path(start_square, goal_square, start_timestep, reservation_table, GRID_SIZE)
        SIMULATION_STEPS = 20  # Define the total simulation steps
     
        if path is not None:
            for step in path:
                # save the path for the agent
                agent.path.append(step[0])
            # Reserve the planned path so that subsequent planning avoids these (position, timestep)

            reserve_path(path, reservation_table)
            # Extend the reservation beyond the final timestep so that the destination remains booked.
            final_position, final_timestep = path[-1]
            # Reserve the final position for all remaining timesteps until SIMULATION_STEPS
            for t in range(final_timestep + 1, SIMULATION_STEPS):
                reservation_table.add((final_position, t))
                # Also, extend the agent's path so that the agent remains at the destination
                agent.path.append(final_position)
        else:
            print("No valid path found.")

    for i in range(0,20):
        plotState(GRID,AgentList,Agent_Assignemnt)
        for agent in AgentList:
            if i < len(agent.path):
                agent.x = agent.path[i][0]
                agent.y = agent.path[i][1]
            else:
                # Use the last position in the path if i exceeds path length
                agent.x = agent.path[-1][0]
                agent.y = agent.path[-1][1]
        print(i)
        time.sleep(1)

####
