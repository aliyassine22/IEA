import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# Constants
GRID_SIZE = 10  # The grid is a 10x10 square
AGENT_COUNT = 20  # Number of agents
FORBIDDEN_DESTINATION = (4, 3)  # The coordinate (4,3) should not be assigned to any agent

# ----------------------- AGENT CLASS -----------------------
class Agent:
    """Represents an agent in the grid world."""
    
    def __init__(self, x, y, agent_id):
        """Initialize agent with position (x, y) and unique ID."""
        self.x = x  # Agent's current x-coordinate
        self.y = y  # Agent's current y-coordinate
        self.agent_id = agent_id  # Unique identifier
        self.target_block = None  # Assigned target block
    
    def calculate_manhattan_distances(self, allowed_blocks):
        """Calculate Manhattan distance to each allowed block and store in a max heap."""
        pq = []
        for (dx, dy) in allowed_blocks:
            if (dx, dy) == FORBIDDEN_DESTINATION:  # Skip forbidden destination
                continue
            distance = abs(self.x - dx) + abs(self.y - dy)
            heapq.heappush(pq, (-distance, (dx, dy)))  # Negative distance for max heap behavior
        return pq

    def assign_destination(self, occupied_positions, allowed_blocks, diamond_positions):
        """
        Assign the farthest available block within the agent's designated half.
        If no block is available in that half, fall back to any available diamond block.
        """
        # Try to assign from the allowed half first.
        pq = self.calculate_manhattan_distances(allowed_blocks)
        while pq:
            _, destination = heapq.heappop(pq)
            if destination not in occupied_positions and destination in diamond_positions:
                self.target_block = destination
                occupied_positions.add(destination)
                print(f"Agent {self.agent_id} assigned to destination {self.target_block}")
                return
        
        # Fallback: if no destination was found in allowed_blocks, search all diamond positions.
        fallback_pq = []
        for (dx, dy) in diamond_positions:
            if (dx, dy) == FORBIDDEN_DESTINATION:
                continue
            distance = abs(self.x - dx) + abs(self.y - dy)
            heapq.heappush(fallback_pq, (-distance, (dx, dy)))
        while fallback_pq:
            _, destination = heapq.heappop(fallback_pq)
            if destination not in occupied_positions:
                self.target_block = destination
                occupied_positions.add(destination)
                print(f"Agent {self.agent_id} fallback assigned to destination {self.target_block}")
                return
        
        print(f"Agent {self.agent_id} could not find an available destination.")

    def move_one_step(self, occupied_positions):
        """Move one step towards the target, prioritizing diagonal movement."""
        if self.target_block and (self.x, self.y) != self.target_block:
            target_x, target_y = self.target_block
            
            # Compute the step direction (-1, 0, or 1)
            dx = np.sign(target_x - self.x)
            dy = np.sign(target_y - self.y)
            
            # Safely remove current position from occupied_positions if present.
            if (self.x, self.y) in occupied_positions:
                occupied_positions.remove((self.x, self.y))
            
            # Prioritize diagonal movement if the cell is free.
            if dx != 0 and dy != 0 and (self.x + dx, self.y + dy) not in occupied_positions:
                self.x += dx
                self.y += dy
            # If diagonal is not available, try moving along the x-axis.
            elif dx != 0 and (self.x + dx, self.y) not in occupied_positions:
                self.x += dx
            # Lastly, try moving along the y-axis.
            elif dy != 0 and (self.x, self.y + dy) not in occupied_positions:
                self.y += dy
            
            # Mark the new position as occupied.
            occupied_positions.add((self.x, self.y))

# ----------------------- CREATE AGENTS FUNCTION -----------------------
def create_agents():
    """Initialize 20 agents in the bottom two rows with unique positions."""
    agent_positions = set()
    agents = []
    while len(agents) < AGENT_COUNT:
        x = np.random.randint(0, 2)  # Only in rows 0 and 1
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) not in agent_positions:
            agent_positions.add((x, y))
            agents.append(Agent(x, y, len(agents)))
    return agents

# ----------------------- GENERATE 20-BLOCK DIAMOND FUNCTION -----------------------
def generate_diamond():
    """Generate a diamond shape with exactly 20 blocks, avoiding the bottom 2 rows."""
    center_x, center_y = 6, 5  # Center of the diamond
    diamond_positions = set()
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (-1, 2), (1, -2), (1, 2)
    ]
    for dx, dy in offsets:
        x, y = center_x + dx, center_y + dy
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) != FORBIDDEN_DESTINATION:
            diamond_positions.add((x, y))
    sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])  # Sort by descending x-coordinate
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, diamond_positions

# ----------------------- STREAMLIT SESSION INITIALIZATION -----------------------
if "agents" not in st.session_state:
    st.session_state.agents = create_agents()

if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
    st.session_state.diamond_upper_half, st.session_state.diamond_lower_half, st.session_state.diamond_positions = generate_diamond()

# ----------------------- GET OCCUPIED POSITIONS FUNCTION -----------------------
def get_occupied_positions():
    """Returns a set of positions occupied by agents."""
    return {(agent.x, agent.y) for agent in st.session_state.agents}

# ----------------------- ASSIGN DESTINATION BLOCKS TO AGENTS -----------------------
occupied_positions = get_occupied_positions()
for agent in st.session_state.agents:
    if agent.x == 1:  # Higher row agents
        agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half, st.session_state.diamond_positions)
    else:  # Lower row agents
        agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half, st.session_state.diamond_positions)

# ----------------------- GRID VISUALIZATION FUNCTION -----------------------
grid_placeholder = st.empty()

def plot_grid():
    """Plots the grid and updates the placeholder in Streamlit for live animation."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    # Plot the diamond shape (gray squares)
    for x, y in st.session_state.diamond_positions:
        ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))
    # Plot agents as red circles with their IDs
    for agent in st.session_state.agents:
        ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
        ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    grid_placeholder.pyplot(fig)

# ----------------------- DISPLAY INITIAL GRID -----------------------
plot_grid()

# ----------------------- MOVE AGENTS ONE BY ONE -----------------------
if st.button("Move Next Agent"):
    occupied_positions = get_occupied_positions()
    for agent in sorted(st.session_state.agents, key=lambda a: -a.x):
        while (agent.x, agent.y) != agent.target_block:
            agent.move_one_step(occupied_positions)
            plot_grid()
            time.sleep(0.3)

st.write("**Agents are moving one by one towards their targets!**")
