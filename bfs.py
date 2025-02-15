import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from collections import deque

# Constants
GRID_SIZE = 10  # The grid is a 10x10 square
AGENT_COUNT = 20  # Number of agents
FORBIDDEN_DESTINATION = (4, 3)  # The coordinate (4,3) should not be assigned to any agent

# Directions (8-connected / Moore neighborhood):
# up, down, left, right, plus 4 diagonals
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

# ----------------------- HELPER FUNCTIONS -----------------------
def in_bounds(x, y):
    """Check if (x, y) is inside the grid and not the forbidden cell."""
    if (x, y) == FORBIDDEN_DESTINATION:
        return False
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def bfs_path(start, goal):
    """
    Returns a list of coordinates forming a path from start to goal using BFS.
    Ignores collisions with other agents; only checks grid bounds and forbidden cell.
    If no path is found, returns an empty list.
    """
    if start == goal:
        return [start]

    queue = deque([start])
    visited = set([start])
    parent = dict()  # to reconstruct the path

    while queue:
        cx, cy = queue.popleft()
        # Check neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if in_bounds(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))
                if (nx, ny) == goal:
                    # Reconstruct path
                    path = []
                    cur = (nx, ny)
                    while cur != start:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(start)
                    path.reverse()
                    return path
    # If we exhaust BFS without finding goal
    return []

# ----------------------- AGENT CLASS -----------------------
class Agent:
    """Represents an agent in the grid world."""
    
    def __init__(self, x, y, agent_id):
        self.x = x               # Agent's current x-coordinate
        self.y = y               # Agent's current y-coordinate
        self.agent_id = agent_id # Unique identifier
        self.target_block = None # Assigned target block
        self.path = []           # Full BFS path to the target

    def calculate_manhattan_distances(self, allowed_blocks):
        """Calculate Manhattan distance to each allowed block and store in a max heap."""
        pq = []
        for (dx, dy) in allowed_blocks:
            if (dx, dy) == FORBIDDEN_DESTINATION:
                continue
            distance = abs(self.x - dx) + abs(self.y - dy)
            # Negative distance for max-heap
            heapq.heappush(pq, (-distance, (dx, dy)))
        return pq

    def assign_destination(self, occupied_positions, allowed_blocks, diamond_positions):
        """
        Assign the farthest available block within the agent's designated half.
        If no block is available in that half, fall back to any available diamond block.
        """
        # 1) Try from the allowed half first
        pq = self.calculate_manhattan_distances(allowed_blocks)
        while pq:
            _, destination = heapq.heappop(pq)
            if destination not in occupied_positions and destination in diamond_positions:
                self.target_block = destination
                occupied_positions.add(destination)
                print(f"Agent {self.agent_id} assigned to destination {self.target_block}")
                return
        
        # 2) Fallback: search all diamond positions
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

    def plan_path(self):
        """Use BFS to plan the path from the agent's current position to the assigned target."""
        if self.target_block:
            self.path = bfs_path((self.x, self.y), self.target_block)
            if not self.path:
                print(f"Agent {self.agent_id} could not find a BFS path to {self.target_block}")

    def move_one_step_along_path(self, occupied_positions):
        """
        Moves the agent along its BFS path (one step at a time).
        This ignores dynamic collisions. If the next cell in the path is blocked,
        the agent will occupy it anyway for demonstration. 
        """
        if not self.path:
            return  # Nothing to do
        
        current_pos = (self.x, self.y)
        # Remove current position from the occupied set
        if current_pos in occupied_positions:
            occupied_positions.remove(current_pos)

        # Move to the next cell in the BFS path
        next_cell = self.path.pop(0)
        self.x, self.y = next_cell

        # Mark new position as occupied
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
    center_x, center_y = 6, 5
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
        if in_bounds(x, y):
            diamond_positions.add((x, y))
    # Sort by descending x-coordinate, then split into top/bottom 10
    sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, diamond_positions

# ----------------------- STREAMLIT SESSION INIT -----------------------
if "agents" not in st.session_state:
    st.session_state.agents = create_agents()

if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
    st.session_state.diamond_upper_half, st.session_state.diamond_lower_half, st.session_state.diamond_positions = generate_diamond()

# ----------------------- GET OCCUPIED POSITIONS FUNCTION -----------------------
def get_occupied_positions():
    return {(agent.x, agent.y) for agent in st.session_state.agents}

# ----------------------- ASSIGN DESTINATION & PLAN BFS PATH -----------------------
occupied_positions = get_occupied_positions()

for agent in st.session_state.agents:
    # Assign a destination cell (farthest in its half or fallback)
    if agent.x == 1:  # row 1 agents -> upper half
        agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half, st.session_state.diamond_positions)
    else:             # row 0 agents -> lower half
        agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half, st.session_state.diamond_positions)
    # Use BFS to plan a path from (agent.x, agent.y) to agent.target_block
    agent.plan_path()

# ----------------------- GRID VISUALIZATION -----------------------
grid_placeholder = st.empty()

def plot_grid():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)

    # Diamond shape
    for x, y in st.session_state.diamond_positions:
        ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))

    # Plot agents
    for ag in st.session_state.agents:
        ax.scatter(ag.y + 0.5, ag.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
        ax.text(ag.y + 0.5, ag.x + 0.5, str(ag.agent_id), color="white",
                ha='center', va='center', fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    grid_placeholder.pyplot(fig)

# Initial display
plot_grid()

# ----------------------- MOVE AGENTS (STEP BY STEP ALONG BFS PATH) -----------------------
if st.button("Move Next Agent"):
    occupied_positions = get_occupied_positions()
    # Move agents from top to bottom (like your original code)
    for ag in sorted(st.session_state.agents, key=lambda a: -a.x):
        # Step along BFS path until the agent has no path left
        while ag.path:
            ag.move_one_step_along_path(occupied_positions)
            plot_grid()
            time.sleep(0.3)

st.write("**Agents are moving one by one along BFS-planned paths!**")
