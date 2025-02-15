import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time

# ----------------------- CONSTANTS -----------------------
GRID_SIZE = 10
AGENT_COUNT = 20
FORBIDDEN_DESTINATION = (4, 3)
DIAMOND_CENTER = (6, 5)  # For layering (used in ordering agents)

# Predefined return targets: fill row 0 first, then row 1.
ROW0_TARGETS = [(0, y) for y in range(GRID_SIZE)]
ROW1_TARGETS = [(1, y) for y in range(GRID_SIZE)]

# ----------------------- DIAMOND PARTITIONS -----------------------
# Diamond is partitioned into an upper and lower half.
# Upper half will be used for agents in row 1.
# Lower half will be used for agents in row 0.
# (They are computed in generate_diamond().)

# 8-connected neighbors (Moore neighborhood)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

# ----------------------- HELPER FUNCTIONS -----------------------
def in_bounds(x, y):
    if (x, y) == FORBIDDEN_DESTINATION:
        return False
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def manhattan_distance(a, b):
    (x1, y1), (x2, y2) = a, b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star_path_with_obstacles(start, goal, obstacles):
    if start == goal:
        return [start]
    open_set = []
    heapq.heappush(open_set, (0, start))
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        closed_set.add(current)
        cx, cy = current
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if not in_bounds(nx, ny) or neighbor in obstacles:
                continue
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def layer_from_center(x, y):
    return manhattan_distance((x, y), DIAMOND_CENTER)

# ----------------------- AGENT CLASS -----------------------
class Agent:
    def __init__(self, x, y, agent_id):
        self.x = x
        self.y = y
        self.agent_id = agent_id
        self.target_block = None  # Diamond target
        self.path = []          # Path to diamond target
        self.return_target = None  # Return target (assigned later)
        self.return_path = []      # Path for return
        self.start_pos = (x, y)    # Original start (not used for return now)
    
    def calculate_manhattan_distances(self, allowed_blocks):
        pq = []
        for (dx, dy) in allowed_blocks:
            if (dx, dy) == FORBIDDEN_DESTINATION:
                continue
            dist = abs(self.x - dx) + abs(self.y - dy)
            heapq.heappush(pq, (-dist, (dx, dy)))
        return pq

    def assign_destination(self, occupied_positions, allowed_blocks, diamond_positions):
        pq = self.calculate_manhattan_distances(allowed_blocks)
        while pq:
            _, candidate = heapq.heappop(pq)
            if candidate not in occupied_positions and candidate in diamond_positions:
                self.target_block = candidate
                occupied_positions.add(candidate)
                print(f"Agent {self.agent_id} -> target {self.target_block}")
                return
        fallback_pq = []
        for (dx, dy) in diamond_positions:
            if (dx, dy) == FORBIDDEN_DESTINATION:
                continue
            dist = abs(self.x - dx) + abs(self.y - dy)
            heapq.heappush(fallback_pq, (-dist, (dx, dy)))
        while fallback_pq:
            _, candidate = heapq.heappop(fallback_pq)
            if candidate not in occupied_positions:
                self.target_block = candidate
                occupied_positions.add(candidate)
                print(f"Agent {self.agent_id} fallback -> target {self.target_block}")
                return
        print(f"Agent {self.agent_id} could not find a diamond target.")

    def plan_path_to_destination(self, obstacles):
        if self.target_block:
            self.path = a_star_path_with_obstacles((self.x, self.y), self.target_block, obstacles)
            if not self.path:
                print(f"[Agent {self.agent_id}] No path to destination {self.target_block}")

    def plan_return_path(self, obstacles):
        if self.return_target:
            self.return_path = a_star_path_with_obstacles((self.x, self.y), self.return_target, obstacles)
            if not self.return_path:
                print(f"[Agent {self.agent_id}] No return path to {self.return_target}")

    def move_one_step(self, occupied_positions, path_key='path'):
        path = getattr(self, path_key, [])
        if not path:
            return
        current_pos = (self.x, self.y)
        if current_pos in occupied_positions:
            occupied_positions.remove(current_pos)
        next_cell = path.pop(0)
        self.x, self.y = next_cell
        occupied_positions.add((self.x, self.y))

# ----------------------- CREATION & DIAMOND GENERATION -----------------------
def create_agents():
    agent_positions = set()
    agents = []
    while len(agents) < AGENT_COUNT:
        x = np.random.randint(0, 2)  # row 0 or row 1
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) not in agent_positions:
            agent_positions.add((x, y))
            agents.append(Agent(x, y, len(agents)))
    return agents

def generate_diamond():
    center_x, center_y = DIAMOND_CENTER
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
    sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, diamond_positions

def get_occupied_positions():
    return {(agent.x, agent.y) for agent in st.session_state.agents}

# ----------------------- STREAMLIT STATE INIT -----------------------
if "agents" not in st.session_state:
    st.session_state.agents = create_agents()

if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
    (st.session_state.diamond_upper_half,
     st.session_state.diamond_lower_half,
     st.session_state.diamond_positions) = generate_diamond()

if "assigned" not in st.session_state:
    st.session_state.assigned = False
if "gone_to_dest" not in st.session_state:
    st.session_state.gone_to_dest = False

grid_placeholder = st.empty()

def plot_grid():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    for i in range(GRID_SIZE+1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    for x, y in st.session_state.diamond_positions:
        ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))
    for ag in st.session_state.agents:
        ax.scatter(ag.y + 0.5, ag.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
        ax.text(ag.y + 0.5, ag.x + 0.5, str(ag.agent_id), color="white", ha="center", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    grid_placeholder.pyplot(fig)

# ----------------------- GO TO DESTINATION (NO COLLISION) -----------------------
def assign_and_go_to_destination():
    occupied_positions = get_occupied_positions()
    # Process agents in two groups:
    # 1. Agents in row 1 (upper row) use the diamond_upper_half.
    row1_agents = [ag for ag in st.session_state.agents if ag.x == 1]
    for agent in row1_agents:
        agent.assign_destination(occupied_positions,
                                 st.session_state.diamond_upper_half,
                                 st.session_state.diamond_positions)
    # 2. Agents in row 0 (lower row) use the diamond_lower_half.
    row0_agents = [ag for ag in st.session_state.agents if ag.x == 0]
    for agent in row0_agents:
        agent.assign_destination(occupied_positions,
                                 st.session_state.diamond_lower_half,
                                 st.session_state.diamond_positions)
    # Move row1 agents first, then row0 agents.
    for group in (row1_agents, row0_agents):
        # Process agents in group in order (outermost first)
        sorted_agents = sorted(group, key=lambda ag: layer_from_center(ag.x, ag.y), reverse=True)
        for ag in sorted_agents:
            obstacles = get_occupied_positions() - {(ag.x, ag.y)}
            ag.plan_path_to_destination(obstacles)
            while ag.path:
                occupied_positions = get_occupied_positions()
                ag.move_one_step(occupied_positions, path_key='path')
                plot_grid()
                time.sleep(0.15)
    st.session_state.gone_to_dest = True

# ----------------------- RETURN TO ORIGINAL (ROW-BASED) -----------------------
def assign_return_targets():
    available_row0 = ROW0_TARGETS.copy()
    available_row1 = ROW1_TARGETS.copy()
    # Process agents in outer-to-inner order.
    sorted_agents = sorted(st.session_state.agents, key=lambda ag: layer_from_center(ag.x, ag.y), reverse=True)
    for ag in sorted_agents:
        if available_row0:
            ag.return_target = available_row0.pop(0)
        else:
            ag.return_target = available_row1.pop(0)
        print(f"Agent {ag.agent_id} return target set to {ag.return_target}")

def return_to_original_no_collision():
    assign_return_targets()
    sorted_agents = sorted(st.session_state.agents, key=lambda ag: layer_from_center(ag.x, ag.y), reverse=True)
    for ag in sorted_agents:
        obstacles = get_occupied_positions() - {(ag.x, ag.y)}
        ag.plan_return_path(obstacles)
        if not ag.return_path:
            print(f"Agent {ag.agent_id}: no return path found; skipping.")
            continue
        while ag.return_path:
            occupied_positions = get_occupied_positions()
            ag.move_one_step(occupied_positions, path_key='return_path')
            plot_grid()
            time.sleep(0.15)

# ----------------------- STREAMLIT UI -----------------------
plot_grid()
col1, col2 = st.columns(2)
with col1:
    if st.button("Go to Destination (No Collision)"):
        assign_and_go_to_destination()
with col2:
    if st.button("Return to Original (Row 0 then 1, No Collision)"):
        if not st.session_state.gone_to_dest:
            st.warning("Agents haven't reached the diamond. Returning from current positions.")
        return_to_original_no_collision()
st.write("**Press 'Go to Destination' to have agents fill the diamond (row 1 then row 0) without collisions.**")
st.write("**Press 'Return to Original' to have agents return to row 0 then row 1 sequentially without collisions.**")
