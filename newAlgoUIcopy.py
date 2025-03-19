import heapq
import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

# Import helper functions and classes from your module.
# (These include Agent, Environment, plot_shape, and plot_agents.)
from greedyWeightMapUI import Agent, Environment, plot_shape, plot_agents

# -------------------------
# A* Planning & Reservation
# -------------------------
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
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def movement_cost(current, neighbor):
    return 1

def get_neighbors(position):
    x, y = position
    # Eight directions (cardinal and diagonal)
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1),
            (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
    
def is_valid(position, grid_size):
    x, y = position
    return 0 <= x < grid_size and 0 <= y < grid_size

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append((node.position, node.timestep))
        node = node.parent
    path.reverse()
    return path

def plan_path(start_square, goal_square, start_timestep, occupied_set, grid_size):
    open_list = []
    closed_set = {}  # key: (position, timestep) -> Node
    
    start_node = Node(start_square, start_timestep, 0, heuristic(start_square, goal_square))
    heapq.heappush(open_list, start_node)
    closed_set[(start_square, start_timestep)] = start_node

    while open_list:
        current = heapq.heappop(open_list)
        if current.position == goal_square:
            return reconstruct_path(current)
        for neighbor in get_neighbors(current.position):
            next_timestep = current.timestep + 1
            if not is_valid(neighbor, grid_size):
                continue
            if (neighbor, next_timestep) in occupied_set:
                continue
            tentative_g = current.g + movement_cost(current.position, neighbor)
            neighbor_key = (neighbor, next_timestep)
            if neighbor_key not in closed_set or tentative_g < closed_set[neighbor_key].g:
                new_node = Node(neighbor, next_timestep, tentative_g,
                                heuristic(neighbor, goal_square), current)
                heapq.heappush(open_list, new_node)
                closed_set[neighbor_key] = new_node

    return None

def reserve_path(path, occupied_set):
    for square, timestep in path:
        occupied_set.add((square, timestep))

# -------------------------
# Lane-Based Assignment (from your second code)
# -------------------------
def splitagents(agentList):
    upperList = []
    lowerList = []
    for agent in agentList:
        if agent.y == 1:
            upperList.append(agent)
        else:
            lowerList.append(agent)
    return upperList, lowerList

def assignDestInLane(agentList, shape_positions):
    assignments = {}
    upperList, lowerList = splitagents(agentList)
    # Create a lane_map: key = lane (x-coordinate), value = a heap of y values (stored as negative for max-heap behavior)
    lane_map = {}
    for x, y in shape_positions:
        if x not in lane_map:
            lane_map[x] = []
        heapq.heappush(lane_map[x], -y)
    # Track available slots per lane
    arr = np.zeros(10, dtype=int)
    for x in lane_map:
        arr[x] = len(lane_map[x])
    
    # First try to assign agents in the same lane as their x-coordinate
    for agent in upperList[:]:
        for i in range(len(arr)):
            if i == agent.x and i in lane_map and len(lane_map[i]) > 0:
                assignments[agent] = (i, -heapq.heappop(lane_map[i]))
                arr[i] -= 1
                upperList.remove(agent)
                break
    for agent in lowerList[:]:
        for i in range(len(arr)):
            if i == agent.x and i in lane_map and len(lane_map[i]) > 0:
                assignments[agent] = (i, -heapq.heappop(lane_map[i]))
                arr[i] -= 1
                lowerList.remove(agent)
                break
    # For remaining agents, assign to the closest available lane based on agent.x
    for agent in upperList:
        minDiff = float('inf')
        closest_x = -1
        for i in range(10):
            if i in lane_map and arr[i] > 0 and abs(agent.x - i) < minDiff:
                minDiff = abs(agent.x - i)
                closest_x = i
        if closest_x != -1:
            assignments[agent] = (closest_x, -heapq.heappop(lane_map[closest_x]))
            arr[closest_x] -= 1
    for agent in lowerList:
        minDiff = float('inf')
        closest_x = -1
        for i in range(10):
            if i in lane_map and arr[i] > 0 and abs(agent.x - i) < minDiff:
                minDiff = abs(agent.x - i)
                closest_x = i
        if closest_x != -1:
            assignments[agent] = (closest_x, -heapq.heappop(lane_map[closest_x]))
            arr[closest_x] -= 1

    return assignments

# -------------------------
# Custom Plot Function (UI like the second code)
# -------------------------
def plotState_custom(environment, agentList, agent_assignment, GRID_SIZE=10):
    agent_positions = [(agent.x, agent.y) for agent in agentList]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    # Plot the destination shape
    plot_shape(ax, environment.shape_positions, GRID_SIZE)
    # Plot agents
    plot_agents(ax, agent_positions, GRID_SIZE)
    # Draw text for agent IDs at both current positions and destination positions
    for agent, destination in agent_assignment.items():
        ax.text(agent.x+0.5, agent.y+0.5, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
        # ax.text(destination[0]+0.5, destination[1]+0.5, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
    if hasattr(environment, 'barriers'):
        plot_barriers(ax, environment.barriers, GRID_SIZE)
    plt.close(fig)
    return fig
def plot_barriers(ax, barrier_positions, GRID_SIZE=10):
    for x, y in barrier_positions:
        rect = plt.Rectangle((x , y ), 1, 1, color='black')
        ax.add_patch(rect)
# -------------------------
# Streamlit Simulation UI
# -------------------------
def run_simulation():
    st.title("A* Path Planning Simulation with Lane-Based Assignment")
    st.write("This simulation uses A* search for path planning and a lane-based assignment for agents to destination cells.")
    
    # Allow the user to select the destination shape.
    shape_type = st.selectbox("Select the Shape", ["diamond", "rectangle", "cross"], index=0)
    barriers_flag = st.selectbox("Barriers", ["True","False"], index=0)

    # Initialize session state variables when shape changes or on first run.
    if 'grid' not in st.session_state or st.session_state.selected_shape != shape_type or st.session_state.barriersBool != barriers_flag:
        st.session_state.selected_shape = shape_type
        st.session_state.barriersBool= barriers_flag  # store selected shape
        st.session_state.grid = Environment(shape=shape_type,barriers_generation=True if st.session_state.barriersBool=="True" else False)

        agents = []
        i = 0
        for y in range(2):  # Create agents in 2 rows × 10 columns.
            for x in range(10):
                agents.append(Agent(x, 1 - y, i))
                i += 1
        if len(st.session_state.grid.shape_positions) < len(agents):
            agents = agents[:len(st.session_state.grid.shape_positions)]
        st.session_state.agents = agents
        
        # Use lane-based assignment instead of Hungarian algorithm.
        st.session_state.assignment = assignDestInLane(agents, st.session_state.grid.shape_positions)
        st.session_state.iter = 0
        st.session_state.simulation_steps = 20
        st.session_state.reservation_table = set()
        if(hasattr(st.session_state.grid, 'barriers')):
            for barrier in st.session_state.grid.barriers:
                for i in range(st.session_state.simulation_steps):
                    st.session_state.reservation_table.add(((barrier[0], barrier[1]), i))

        # For each agent, plan its path from start to destination.
        for agent in st.session_state.agents:
            start_square = (agent.x, agent.y)
            goal_square = st.session_state.assignment[agent]
            start_timestep = 0
            path = plan_path(start_square, goal_square, start_timestep, st.session_state.reservation_table, st.session_state.grid.GRID_SIZE)
            if path is not None:
                # Save only positions (ignoring timesteps)
                for step in path:
                    agent.path.append(step[0])
                reserve_path(path, st.session_state.reservation_table)
                final_position, final_timestep = path[-1]
                # Extend the reservation so the destination remains booked.
                for t in range(final_timestep + 1, st.session_state.simulation_steps):
                    st.session_state.reservation_table.add((final_position, t))
                    agent.path.append(final_position)
            else:
                st.write(f"No valid path found for agent {agent.agent_id}")
    
    # Placeholders for the plot and iteration counter.
    plot_placeholder = st.empty()
    iter_placeholder = st.empty()
    
    # Button to show the initial grid.
    if st.button("Show Grid"):
        st.session_state.iter = 0
        fig = plotState_custom(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)
        iter_placeholder.write(f"Step: {st.session_state.iter}")
    
    # Button for a single iteration.
    if st.button("Next Iteration"):
        # Stop if all agents are at their destination.
        if all((agent.x, agent.y) == agent.path[-1] for agent in st.session_state.agents):
            st.write("All agents have reached their destination. Simulation stopped.")
        else:
            next_step = st.session_state.iter + 1
            for agent in st.session_state.agents:
                if next_step < len(agent.path):
                    agent.x, agent.y = agent.path[next_step]
                else:
                    agent.x, agent.y = agent.path[-1]
            st.session_state.iter = next_step
        iter_placeholder.write(f"Step: {st.session_state.iter}")
        fig = plotState_custom(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)
    
    # Button to run the simulation continuously.
    if st.button("Run Simulation"):
        st.session_state.running = True
        while st.session_state.running:
            if all((agent.x, agent.y) == agent.path[-1] for agent in st.session_state.agents):
                st.write("All agents have reached their destination. Simulation stopped.")
                break
            next_step = st.session_state.iter + 1
            for agent in st.session_state.agents:
                if next_step < len(agent.path):
                    agent.x, agent.y = agent.path[next_step]
                else:
                    agent.x, agent.y = agent.path[-1]
            st.session_state.iter = next_step
            iter_placeholder.write(f"Step: {st.session_state.iter}")
            fig = plotState_custom(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
            plot_placeholder.pyplot(fig)
            time.sleep(1)
        st.session_state.running = False
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")
    
    # Button to stop the simulation manually.
    if st.button("Stop Simulation"):
        st.session_state.running = False
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")
        fig = plotState_custom(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)

if __name__ == "__main__":
    run_simulation()
