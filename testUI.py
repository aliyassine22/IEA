import heapq
import time
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

# Importing helper functions and classes from your original module.
# (These include Agent, Environment, hungarianAssignment, plot_shape, and plot_agents.)
from greedyWeightMapUI import Agent, Environment, hungarianAssignment, plot_shape, plot_agents

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
    # Use Manhattan distance as heuristic
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def movement_cost(current, neighbor):
    # For grid movement with cardinal moves, cost is constant (1)
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
    """
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
    """
    Reserves the path in the occupied_set.
    Each element in the path is a tuple (position, timestep).
    """
    for square, timestep in path:
        occupied_set.add((square, timestep))

# -------------------------
# UI Plotting Functions
# -------------------------
def plot_destinations(ax, agent_assignments, GRID_SIZE):
    """
    Draws the agent IDs at their current positions and at their assigned destinations.
    """
    for agent, destination in agent_assignments.items():
        # Plot agent ID at current position
        ax.text(agent.x + 0.5, agent.y + 0.5, str(agent.agent_id),
                ha='center', va='center', fontsize=8, color='black')
        # Plot the same ID at the destination position
        # ax.text(destination[0] + 0.5, destination[1] + 0.5, str(agent.agent_id),
        #         ha='center', va='center', fontsize=8, color='black')

def plotState(environment, agentList, agent_assignment, GRID_SIZE=10):
    """
    Plots the current state of the environment, agents, and destination assignments.
    (Weights are not plotted.)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    # Draw destination cells (shape)
    plot_shape(ax, environment.shape_positions, GRID_SIZE)
    # Draw agent positions as circles
    agent_positions = [(agent.x, agent.y) for agent in agentList]
    plot_agents(ax, agent_positions, GRID_SIZE)
    # Draw text for agent IDs and their destinations
    plot_destinations(ax, agent_assignment, GRID_SIZE)
    
    ax.set_aspect('equal', adjustable='box')
    return fig

# -------------------------
# Streamlit Simulation UI
# -------------------------
def run_simulation():
    st.title("A* Path Planning Simulation")
    st.write("This simulation uses A* search for path planning and the Hungarian algorithm for agent-destination assignment.")
    
    # Allow the user to select the shape of the destination region.
    shape_type = st.selectbox("Select the Shape", ["diamond", "rectangle", "cross"], index=0)
    
    # Initialize session state variables when shape selection changes or on first run.
    if 'grid' not in st.session_state or st.session_state.selected_shape != shape_type:
        st.session_state.selected_shape = shape_type
        st.session_state.grid = Environment(shape=shape_type)
        agents = []
        i = 0
        for y in range(2):  # Create agents arranged in 2 rows x 10 columns.
            for x in range(10):
                agents.append(Agent(x, 1 - y, i))
                i += 1
        # If there are more agents than available destination cells, trim the list.
        if len(st.session_state.grid.shape_positions) < len(agents):
            agents = agents[:len(st.session_state.grid.shape_positions)]
        st.session_state.agents = agents
        
        # Compute agent-to-destination assignment using the Hungarian algorithm.
        st.session_state.assignment = hungarianAssignment(agents, st.session_state.grid.shape_positions)
        st.session_state.iter = 0
        st.session_state.simulation_steps = 20
        st.session_state.reservation_table = set()
        
        # For each agent, plan its path from its start position to its assigned destination.
        for agent in st.session_state.agents:
            start_square = (agent.x, agent.y)
            goal_square = st.session_state.assignment[agent]
            start_timestep = 0
            path = plan_path(start_square, goal_square, start_timestep, st.session_state.reservation_table, st.session_state.grid.GRID_SIZE)
            if path is not None:
                # Save only the positions (ignore timesteps) in the agent's path.
                for step in path:
                    agent.path.append(step[0])
                reserve_path(path, st.session_state.reservation_table)
                final_position, final_timestep = path[-1]
                # Extend the reservation beyond the final timestep so the destination remains booked.
                for t in range(final_timestep + 1, st.session_state.simulation_steps):
                    st.session_state.reservation_table.add((final_position, t))
                    agent.path.append(final_position)
            else:
                st.write(f"No valid path found for agent {agent.agent_id}")
    
    # Placeholders for plot and iteration counter
    plot_placeholder = st.empty()
    iter_placeholder = st.empty()
    
    # Button to show the initial grid
    if st.button("Show Grid"):
        st.session_state.iter = 0
        fig = plotState(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)
        iter_placeholder.write(f"Step: {st.session_state.iter}")
    
    # Button for a single iteration (step)
    if st.button("Next Iteration"):
        # First check if all agents are at their destination
        if all((agent.x, agent.y) == agent.path[-1] for agent in st.session_state.agents):
            st.write("All agents have reached their destination. Simulation stopped.")
        else:
            # Compute the next step index only if not all agents are done
            next_step = st.session_state.iter + 1
            for agent in st.session_state.agents:
                if next_step < len(agent.path):
                    agent.x, agent.y = agent.path[next_step]
                else:
                    agent.x, agent.y = agent.path[-1]
            st.session_state.iter = next_step  # Update counter only after agents move
        iter_placeholder.write(f"Step: {st.session_state.iter}")
        fig = plotState(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)


    # Button to run the simulation continuously
    if st.button("Run Simulation"):
        st.session_state.running = True
        while st.session_state.running:
            # Check if agents are already at their destination
            if all((agent.x, agent.y) == agent.path[-1] for agent in st.session_state.agents):
                st.write("All agents have reached their destination. Simulation stopped.")
                break

            next_step = st.session_state.iter + 1
            for agent in st.session_state.agents:
                if next_step < len(agent.path):
                    agent.x, agent.y = agent.path[next_step]
                else:
                    agent.x, agent.y = agent.path[-1]
            st.session_state.iter = next_step  # Increment only if agents have moved
            iter_placeholder.write(f"Step: {st.session_state.iter}")
            fig = plotState(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
            plot_placeholder.pyplot(fig)
            time.sleep(1)
            
        st.session_state.running = False
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")

    # Button to stop the simulation if running
    if st.button("Stop Simulation"):
        st.session_state.running = False
        iter_placeholder.write(f"Simulation stopped at Step: {st.session_state.iter}")
        fig = plotState(st.session_state.grid, st.session_state.agents, st.session_state.assignment, st.session_state.grid.GRID_SIZE)
        plot_placeholder.pyplot(fig)

if __name__ == "__main__":
    run_simulation()
