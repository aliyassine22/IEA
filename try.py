import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from collections import deque

##############################################################################
# GLOBAL CONSTANTS
##############################################################################

GRID_SIZE = 10
AGENT_COUNT = 20
FORBIDDEN_DESTINATION = (4, 3)

# Example 16 boundary cells (update for your actual shape).
BOUNDARY_CELLS = {
    (5,4), (5,5), (5,6), (5,7),
    (6,3), (6,4), (6,5), (6,6), (6,7), (6,8),
    (7,3), (7,8),
    (8,4), (8,5), (8,6), (8,7)
}

# Optional BFS jump portals if you want “teleportation.”
BOUNDARY_PORTALS = {
    (5,4): (5,7),
    (5,7): (5,4),
    (6,3): (6,8),
    (6,8): (6,3),
    (7,3): (7,8),
    (7,8): (7,3),
    # etc.
}

##############################################################################
# 1. BFS UTILITIES
##############################################################################

def bfs_with_jump(start, goal, obstacles, allow_jump=False):
    """
    Standard 8-direction BFS from 'start' to 'goal'.
    If allow_jump=True, boundary cells in BOUNDARY_PORTALS act like “teleports.”
    """
    if start == goal:
        return [start]
    
    queue = deque()
    queue.append((start, [start]))
    visited = {start}
    
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    while queue:
        current, path = queue.popleft()
        
        # Normal BFS expansions
        for dx, dy in directions:
            nxt = (current[0] + dx, current[1] + dy)
            if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE:
                if nxt == goal:
                    return path + [nxt]
                if nxt not in obstacles and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        
        # Jump expansions
        if allow_jump and current in BOUNDARY_PORTALS:
            jump_target = BOUNDARY_PORTALS[current]
            if jump_target and jump_target not in visited:
                visited.add(jump_target)
                new_path = path + [jump_target]
                if jump_target == goal:
                    return new_path
                queue.append((jump_target, new_path))
    
    return None

##############################################################################
# 2. AGENT CLASS
##############################################################################

class Agent:
    def __init__(self, x, y, agent_id):
        self.x = x
        self.y = y
        self.agent_id = agent_id
        self.target_block = None       # assigned destination
        self.boundary_flag = False     # set True once the boundary is formed and agent is on boundary
        self.has_reached_target = False
    
    def is_at_target(self):
        return (self.x, self.y) == self.target_block
    
    def compute_next_move(self, occupied_positions, boundary_formed):
        """
        Compute the next BFS step toward self.target_block.
        If boundary_formed=True, BFS can use jump portals.
        """
        if self.target_block and not self.is_at_target():
            obstacles = occupied_positions.copy()
            obstacles.discard((self.x, self.y))  # so BFS can start from current cell
            path = bfs_with_jump(
                start=(self.x, self.y),
                goal=self.target_block,
                obstacles=obstacles,
                allow_jump=boundary_formed
            )
            if path and len(path) >= 2:
                return path[1]
        return (self.x, self.y)

##############################################################################
# 3. CREATE AGENTS & DIAMOND
##############################################################################

def create_agents():
    agents = []
    occupied = set()
    while len(agents) < AGENT_COUNT:
        x = np.random.randint(0, 2)  # bottom rows
        y = np.random.randint(0, GRID_SIZE)
        if (x, y) not in occupied:
            occupied.add((x, y))
            agents.append(Agent(x, y, len(agents)))
    return agents

def generate_diamond():
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
        xx, yy = center_x + dx, center_y + dy
        if 0 <= xx < GRID_SIZE and 0 <= yy < GRID_SIZE and (xx, yy) != FORBIDDEN_DESTINATION:
            diamond_positions.add((xx, yy))
    
    sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
    upper_half = set(sorted_blocks[:10])
    lower_half = set(sorted_blocks[10:])
    return upper_half, lower_half, diamond_positions

if "agents" not in st.session_state:
    st.session_state.agents = create_agents()

if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
    (st.session_state.diamond_upper_half,
     st.session_state.diamond_lower_half,
     st.session_state.diamond_positions) = generate_diamond()

##############################################################################
# 4. ASSIGN DESTINATIONS (DIVIDE & CONQUER)
##############################################################################

def get_occupied_positions():
    return {(ag.x, ag.y) for ag in st.session_state.agents}

def assign_destinations():
    occupied_dest = set()
    agents_sorted = sorted(st.session_state.agents, key=lambda a: -a.x)
    for ag in agents_sorted:
        if ag.target_block is not None:
            continue
        # Assign from upper half first
        free_upper = st.session_state.diamond_upper_half - occupied_dest
        if free_upper:
            pq = []
            for block in free_upper:
                dist = abs(ag.x - block[0]) + abs(ag.y - block[1])
                heapq.heappush(pq, (-dist, block))
            if pq:
                _, chosen = heapq.heappop(pq)
                ag.target_block = chosen
                occupied_dest.add(chosen)
                continue
        # Otherwise from lower half
        free_lower = st.session_state.diamond_lower_half - occupied_dest
        if free_lower:
            pq = []
            for block in free_lower:
                dist = abs(ag.x - block[0]) + abs(ag.y - block[1])
                heapq.heappush(pq, (-dist, block))
            if pq:
                _, chosen = heapq.heappop(pq)
                ag.target_block = chosen
                occupied_dest.add(chosen)

if "destinations_assigned" not in st.session_state:
    assign_destinations()
    st.session_state.destinations_assigned = True

##############################################################################
# 5. BOUNDARY FORMATION & MOVE-DEEPER LOGIC
##############################################################################

def check_boundary_formed():
    """
    If all BOUNDARY_CELLS are occupied, mark occupant agents as boundary_flag=True.
    Returns True if boundary is fully formed.
    """
    occ = get_occupied_positions()
    if BOUNDARY_CELLS.issubset(occ):
        for ag in st.session_state.agents:
            if (ag.x, ag.y) in BOUNDARY_CELLS:
                ag.boundary_flag = True
        return True
    return False

def find_deeper_cell(agent, occupied_positions):
    """
    Finds any free cell in the diamond that is NOT in BOUNDARY_CELLS
    and that 'agent' can BFS to. Returns that cell if found, else None.
    """
    deeper_candidates = (st.session_state.diamond_positions - BOUNDARY_CELLS) - occupied_positions
    obstacles = occupied_positions.copy()
    obstacles.discard((agent.x, agent.y))
    for cell in deeper_candidates:
        path = bfs_with_jump((agent.x, agent.y), cell, obstacles, allow_jump=True)
        if path:
            return cell
    return None

##############################################################################
# 6. SWAP LOGIC: OUTSIDE AGENT <-> CLOSEST BOUNDARY AGENT
##############################################################################

def swap_with_closest_boundary_agent(outside_agent, boundary_agents, occ):
    """
    Given an outside agent that is blocked, we look at the boundary_agents list
    (all boundary agents currently on boundary cells and at their target).
    1. Sort boundary agents by Manhattan distance from 'outside_agent'.
    2. For each boundary agent in ascending distance, check if it can move deeper.
    3. If yes, we do the swap:
       - boundary agent picks a deeper cell as new target_block
       - outside_agent picks boundary agent's old boundary cell
    4. Return True if a swap was done, else False.
    """
    # Sort boundary agents by distance to the outside agent.
    boundary_agents_sorted = sorted(
        boundary_agents,
        key=lambda ba: abs(ba.x - outside_agent.x) + abs(ba.y - outside_agent.y)
    )
    # Try each boundary agent in ascending distance.
    for b_ag in boundary_agents_sorted:
        deeper_cell = find_deeper_cell(b_ag, occ)
        if deeper_cell:
            old_boundary_cell = (b_ag.x, b_ag.y)
            b_ag.target_block = deeper_cell
            outside_agent.target_block = old_boundary_cell
            return True
    return False

def reassign_blocked_outside_agents():
    """
    For each outside agent that cannot BFS to its original target, we find the closest
    boundary agent that can move deeper, and do the swap.
    We do this for all outside agents in a single pass.
    """
    boundary_formed = check_boundary_formed()
    if not boundary_formed:
        return
    
    occ = get_occupied_positions()
    # Gather boundary agents that are "settled" on the boundary (i.e. is_at_target).
    boundary_agents = [
        ag for ag in st.session_state.agents
        if ag.boundary_flag and (ag.x, ag.y) == ag.target_block
    ]
    if not boundary_agents:
        return
    
    # For each outside agent, check if BFS to target is blocked.
    for outside_ag in st.session_state.agents:
        if outside_ag.is_at_target():
            continue  # already done
        # Try BFS to original target
        obstacles = occ.copy()
        obstacles.discard((outside_ag.x, outside_ag.y))
        path = bfs_with_jump((outside_ag.x, outside_ag.y),
                             outside_ag.target_block,
                             obstacles,
                             allow_jump=True)
        if path:
            continue  # no need to swap
        # BFS is blocked => attempt the swap with a boundary agent
        swapped = swap_with_closest_boundary_agent(outside_ag, boundary_agents, occ)
        if swapped:
            # Once swapped, outside_ag has a new boundary target.
            # We do not do multiple swaps for the same agent in this pass.
            continue

##############################################################################
# 7. PARALLEL MOVEMENT WITH COLLISION RESOLUTION
##############################################################################

grid_placeholder = st.empty()

def plot_grid():
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    
    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)
    
    # Draw diamond shape
    for (dx, dy) in st.session_state.diamond_positions:
        ax.add_patch(plt.Rectangle((dy, dx), 1, 1, color="gray", alpha=0.5))
    
    # Highlight boundary cells
    for (bx, by) in BOUNDARY_CELLS:
        ax.add_patch(plt.Rectangle((by, bx), 1, 1, color="blue", alpha=0.2))
    
    # Plot agents
    for ag in st.session_state.agents:
        color = "red"
        if ag.boundary_flag:
            color = "purple"
        ax.scatter(ag.y + 0.5, ag.x + 0.5, color=color, s=800, marker="o", edgecolors="black")
        ax.text(ag.y + 0.5, ag.x + 0.5, str(ag.agent_id), color="white",
                ha='center', va='center', fontsize=12)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    grid_placeholder.pyplot(fig)

def parallel_move():
    # Reassign outside agents if blocked
    reassign_blocked_outside_agents()
    
    occ = get_occupied_positions()
    boundary_formed = check_boundary_formed()
    
    proposed_moves = {}
    agent_moves = {}
    
    # BFS-based proposals
    for ag in st.session_state.agents:
        if ag.is_at_target():
            agent_moves[ag.agent_id] = (ag.x, ag.y)
        else:
            nxt = ag.compute_next_move(occ, boundary_formed)
            agent_moves[ag.agent_id] = nxt
            proposed_moves.setdefault(nxt, []).append(ag)
    
    # Collision resolution
    allowed_moves = {}
    for cell, contenders in proposed_moves.items():
        if len(contenders) == 1:
            allowed_moves[contenders[0].agent_id] = cell
        else:
            # The agent whose target has higher x gets priority
            contenders.sort(key=lambda a: (a.target_block[0], -a.agent_id), reverse=True)
            winner = contenders[0]
            allowed_moves[winner.agent_id] = cell
            for loser in contenders[1:]:
                allowed_moves[loser.agent_id] = (loser.x, loser.y)
    
    # Update positions
    for ag in st.session_state.agents:
        new_pos = allowed_moves.get(ag.agent_id, (ag.x, ag.y))
        ag.x, ag.y = new_pos
    
    plot_grid()

##############################################################################
# 8. STREAMLIT BUTTON
##############################################################################

if st.button("Move Next Step"):
    parallel_move()
    time.sleep(0.3)

plot_grid()
st.write("**Boundary formation + BFS + closest-boundary-agent swap logic for outside agents!**")



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import heapq
# import time
# from collections import deque

# ##############################################################################
# # GLOBAL CONSTANTS
# ##############################################################################

# GRID_SIZE = 10
# AGENT_COUNT = 20
# FORBIDDEN_DESTINATION = (4, 3)

# # 1) The diamond shape is created as usual, but you can adapt it to circles/squares/triangles.
# # 2) BOUNDARY_CELLS define the ring that eventually forms a “closed” shape.
# # 3) If you want BFS "teleport" jumps across boundary, define BOUNDARY_PORTALS.

# BOUNDARY_CELLS = {
#     # Example placeholders—replace with your actual boundary coordinates (16 total, etc.)
#     (5,4), (5,5), (5,6), (5,7),
#     (6,3), (6,4), (6,5), (6,6), (6,7), (6,8),
#     (7,3), (7,8),
#     (8,4), (8,5), (8,6), (8,7)
# }

# # Optional “jump” pairs: if you want BFS to treat these as “portals.”
# BOUNDARY_PORTALS = {
#     # (left_cell) -> (right_cell), (right_cell) -> (left_cell), etc.
#     # Only needed if you want the BFS jump feature.
#     (5,4): (5,7),
#     (5,7): (5,4),
#     (6,3): (6,8),
#     (6,8): (6,3),
#     (7,3): (7,8),
#     (7,8): (7,3),
#     # ...
# }

# ##############################################################################
# # 1. BFS UTILITIES
# ##############################################################################

# def bfs_with_jump(start, goal, obstacles, allow_jump=False):
#     """
#     Performs BFS from 'start' to 'goal' in 8 directions.
#     If allow_jump=True, boundary cells in BOUNDARY_PORTALS can “teleport” to their paired cell.
#     """
#     if start == goal:
#         return [start]
    
#     queue = deque()
#     queue.append((start, [start]))
#     visited = {start}
    
#     # 8-direction moves
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),           (0, 1),
#                   (1, -1),  (1, 0),  (1, 1)]
    
#     while queue:
#         current, path = queue.popleft()
        
#         # 1) Normal BFS expansions
#         for dx, dy in directions:
#             nxt = (current[0] + dx, current[1] + dy)
#             if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE:
#                 if nxt == goal:
#                     return path + [nxt]
#                 if nxt not in obstacles and nxt not in visited:
#                     visited.add(nxt)
#                     queue.append((nxt, path + [nxt]))
        
#         # 2) Jump expansions (if allowed and we have a portal)
#         if allow_jump and current in BOUNDARY_PORTALS:
#             jump_target = BOUNDARY_PORTALS[current]
#             if jump_target and jump_target not in visited:
#                 visited.add(jump_target)
#                 new_path = path + [jump_target]
#                 if jump_target == goal:
#                     return new_path
#                 queue.append((jump_target, new_path))
    
#     return None  # No path found

# ##############################################################################
# # 2. AGENT CLASS
# ##############################################################################

# class Agent:
#     def __init__(self, x, y, agent_id):
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id
#         self.target_block = None       # assigned destination
#         self.boundary_flag = False     # True if occupying a boundary cell
#         self.has_reached_target = False  # convenience flag if we want to track arrival
    
#     def is_at_target(self):
#         return (self.x, self.y) == self.target_block
    
#     def compute_next_move(self, occupied_positions, boundary_formed):
#         """
#         Returns the next cell along BFS path to self.target_block.
#         If boundary_formed is True, BFS can use jump portals.
#         If agent is already at target, returns current position.
#         """
#         if self.target_block and not self.is_at_target():
#             obstacles = occupied_positions.copy()
#             # remove self so BFS can start from current
#             obstacles.discard((self.x, self.y))
            
#             path = bfs_with_jump(
#                 start=(self.x, self.y),
#                 goal=self.target_block,
#                 obstacles=obstacles,
#                 allow_jump=boundary_formed  # only allow jump BFS if boundary is formed
#             )
#             if path and len(path) >= 2:
#                 return path[1]  # next step
#         return (self.x, self.y)  # no move

# ##############################################################################
# # 3. CREATE AGENTS & DIAMOND SHAPE
# ##############################################################################

# def create_agents():
#     agents = []
#     occupied = set()
#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)  # bottom two rows
#         y = np.random.randint(0, GRID_SIZE)
#         if (x, y) not in occupied:
#             occupied.add((x, y))
#             agents.append(Agent(x, y, len(agents)))
#     return agents

# def generate_diamond():
#     """
#     Example: 20-block diamond shape around center (6,5). 
#     Splits into upper and lower halves for 'divide & conquer' assignment.
#     """
#     center_x, center_y = 6, 5
#     diamond_positions = set()
#     offsets = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2),
#         (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1),
#         (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]
#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) != FORBIDDEN_DESTINATION:
#             diamond_positions.add((x, y))
    
#     # Split into two sets of 10
#     sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
#     upper_half = set(sorted_blocks[:10])
#     lower_half = set(sorted_blocks[10:])
#     return upper_half, lower_half, diamond_positions

# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
#     (st.session_state.diamond_upper_half,
#      st.session_state.diamond_lower_half,
#      st.session_state.diamond_positions) = generate_diamond()

# ##############################################################################
# # 4. ASSIGN DESTINATIONS (DIVIDE & CONQUER)
# ##############################################################################

# def get_occupied_positions():
#     return {(ag.x, ag.y) for ag in st.session_state.agents}

# def assign_destinations():
#     occupied_dest = set()
#     agents_sorted = sorted(st.session_state.agents, key=lambda a: -a.x)
    
#     for ag in agents_sorted:
#         if ag.target_block is not None:
#             continue
#         # try upper half first
#         free_upper = st.session_state.diamond_upper_half - occupied_dest
#         if free_upper:
#             # pick the farthest block
#             pq = []
#             for block in free_upper:
#                 dist = abs(ag.x - block[0]) + abs(ag.y - block[1])
#                 heapq.heappush(pq, (-dist, block))
#             if pq:
#                 _, chosen = heapq.heappop(pq)
#                 ag.target_block = chosen
#                 occupied_dest.add(chosen)
#                 continue
        
#         # otherwise lower half
#         free_lower = st.session_state.diamond_lower_half - occupied_dest
#         if free_lower:
#             pq = []
#             for block in free_lower:
#                 dist = abs(ag.x - block[0]) + abs(ag.y - block[1])
#                 heapq.heappush(pq, (-dist, block))
#             if pq:
#                 _, chosen = heapq.heappop(pq)
#                 ag.target_block = chosen
#                 occupied_dest.add(chosen)

# if "destinations_assigned" not in st.session_state:
#     assign_destinations()
#     st.session_state.destinations_assigned = True

# ##############################################################################
# # 5. BOUNDARY FORMATION & MOVE-DEEPER LOGIC
# ##############################################################################

# def check_boundary_formed():
#     """
#     If all boundary cells are occupied, mark those occupant agents as boundary_flag = True.
#     Returns True if boundary is fully formed, else False.
#     """
#     occ = get_occupied_positions()
#     if BOUNDARY_CELLS.issubset(occ):
#         # mark occupant agents
#         for ag in st.session_state.agents:
#             if (ag.x, ag.y) in BOUNDARY_CELLS:
#                 ag.boundary_flag = True
#         return True
#     return False

# def find_deeper_cell(agent, occupied_positions):
#     """
#     Finds a free cell inside the diamond that is not in the boundary 
#     (or some other condition) to which 'agent' can move.
#     Returns that cell or None if none found.
#     """
#     # Example: any cell in the diamond that is not in BOUNDARY_CELLS and not occupied
#     deeper_candidates = (st.session_state.diamond_positions - BOUNDARY_CELLS) - occupied_positions
    
#     # BFS to see if we can reach any deeper candidate
#     # Return the first reachable one (or pick by distance).
#     obstacles = occupied_positions.copy()
#     obstacles.discard((agent.x, agent.y))
#     for cell in deeper_candidates:
#         path = bfs_with_jump((agent.x, agent.y), cell, obstacles, allow_jump=True)
#         if path:
#             return cell
#     return None

# def reassign_boundary_agents():
#     """
#     If boundary is formed, any boundary agent that is 'at its target' on the boundary 
#     can be forced to pick a deeper cell so that outside agents can eventually take its place.
#     """
#     boundary_formed = check_boundary_formed()
#     if not boundary_formed:
#         return
    
#     occ = get_occupied_positions()
#     for ag in st.session_state.agents:
#         # If agent is on boundary, has reached that boundary cell, we want it to move deeper
#         if ag.boundary_flag and ag.is_at_target():
#             # find a deeper cell
#             deeper_cell = find_deeper_cell(ag, occ)
#             if deeper_cell:
#                 # reassign target
#                 ag.target_block = deeper_cell

# def reassign_outside_agents():
#     """
#     For outside agents that cannot BFS to their original target (blocked by boundary),
#     try to let them swap with boundary occupant if that occupant can move deeper.
#     """
#     boundary_formed = check_boundary_formed()
#     if not boundary_formed:
#         return
    
#     occ = get_occupied_positions()
#     for outside_ag in st.session_state.agents:
#         if outside_ag.is_at_target():
#             continue  # already at destination
#         # Check if BFS to original target is still possible
#         obstacles = occ.copy()
#         obstacles.discard((outside_ag.x, outside_ag.y))
#         path = bfs_with_jump((outside_ag.x, outside_ag.y),
#                              outside_ag.target_block,
#                              obstacles,
#                              allow_jump=True)
#         if path:
#             continue  # no need to reassign if path is open
        
#         # BFS is blocked => attempt to swap with a boundary occupant
#         # We'll do a simple approach: search for boundary occupant in boundary_formed
#         # that can move deeper, then reassign them
#         for boundary_ag in st.session_state.agents:
#             if not boundary_ag.boundary_flag:
#                 continue
#             if (boundary_ag.x, boundary_ag.y) == boundary_ag.target_block:
#                 # occupant is "settled" at boundary
#                 deeper_cell = find_deeper_cell(boundary_ag, occ)
#                 if deeper_cell:
#                     # occupant can move deeper
#                     # occupant changes target to deeper cell
#                     boundary_ag.target_block = deeper_cell
#                     # outside agent changes target to occupant's old cell
#                     outside_ag.target_block = (boundary_ag.x, boundary_ag.y)
#                     return  # we do one swap at a time for simplicity

# ##############################################################################
# # 6. PARALLEL MOVEMENT WITH COLLISION RESOLUTION
# ##############################################################################

# grid_placeholder = st.empty()

# def plot_grid():
#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)
    
#     # draw diamond
#     for (dx, dy) in st.session_state.diamond_positions:
#         ax.add_patch(plt.Rectangle((dy, dx), 1, 1, color="gray", alpha=0.5))
    
#     # highlight boundary
#     for (bx, by) in BOUNDARY_CELLS:
#         ax.add_patch(plt.Rectangle((by, bx), 1, 1, color="blue", alpha=0.2))
    
#     # draw agents
#     for ag in st.session_state.agents:
#         color = "red"
#         if ag.boundary_flag:
#             color = "purple"  # boundary occupant
#         ax.scatter(ag.y + 0.5, ag.x + 0.5, color=color, s=800, marker="o", edgecolors="black")
#         ax.text(ag.y + 0.5, ag.x + 0.5, str(ag.agent_id), color="white",
#                 ha='center', va='center', fontsize=12)
    
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)
#     grid_placeholder.pyplot(fig)

# def parallel_move():
#     # 1) Possibly reassign boundary agents so they move deeper
#     reassign_boundary_agents()
#     # 2) Possibly reassign outside agents if they are blocked
#     reassign_outside_agents()
    
#     # 3) Now do normal BFS-based moves in parallel
#     occ = get_occupied_positions()
#     boundary_formed = check_boundary_formed()
    
#     proposed_moves = {}
#     agent_moves = {}
    
#     for ag in st.session_state.agents:
#         if ag.is_at_target():
#             agent_moves[ag.agent_id] = (ag.x, ag.y)
#         else:
#             nxt = ag.compute_next_move(occ, boundary_formed)
#             agent_moves[ag.agent_id] = nxt
#             proposed_moves.setdefault(nxt, []).append(ag)
    
#     # 4) Collision resolution
#     allowed_moves = {}
#     for cell, contenders in proposed_moves.items():
#         if len(contenders) == 1:
#             allowed_moves[contenders[0].agent_id] = cell
#         else:
#             # The agent whose target has higher x gets priority
#             contenders.sort(key=lambda a: (a.target_block[0], -a.agent_id), reverse=True)
#             winner = contenders[0]
#             allowed_moves[winner.agent_id] = cell
#             for loser in contenders[1:]:
#                 allowed_moves[loser.agent_id] = (loser.x, loser.y)
    
#     # 5) Update positions
#     for ag in st.session_state.agents:
#         new_pos = allowed_moves.get(ag.agent_id, (ag.x, ag.y))
#         ag.x, ag.y = new_pos
    
#     plot_grid()

# ##############################################################################
# # 7. STREAMLIT BUTTON
# ##############################################################################

# if st.button("Move Next Step"):
#     parallel_move()
#     time.sleep(0.3)

# plot_grid()
# st.write("**A comprehensive example: boundary formation, BFS, reassignments, and swaps!**")

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import heapq
# import time
# from collections import deque

# GRID_SIZE = 10
# AGENT_COUNT = 20
# FORBIDDEN_DESTINATION = (4, 3)

# ##############################################################################
# # 1. BOUNDARY / PORTALS / SPECIAL CELLS
# ##############################################################################

# # Example boundary cells (adapt to your actual shape).
# BOUNDARY_CELLS = {
#     (5,4), (5,5), (5,6), (5,7),
#     (6,3), (6,4), (6,5), (6,6), (6,7), (6,8),
#     (7,3), (7,8),
#     (8,4), (8,5), (8,6), (8,7)
# }

# # Example “portal” pairs: a dictionary mapping each boundary cell to the cell
# # on the “other side.” You must fill in the correct pairs for your shape.
# BOUNDARY_PORTALS = {
#     (5,4): (5,7),  # jump from left boundary to right boundary
#     (5,7): (5,4),
#     (6,3): (6,8),
#     (6,8): (6,3),
#     (7,3): (7,8),
#     (7,8): (7,3),
#     # etc. for each boundary pair you want to link
#     # If a cell does not have a “jump,” omit or map it to None
# }

# ##############################################################################
# # 2. BFS WITH OPTIONAL “JUMP”
# ##############################################################################

# def bfs_with_jump(start, goal, obstacles, allow_jump=False):
#     """
#     Perform a BFS from start to goal.
#     If allow_jump=True, then any time we reach a boundary cell that has a portal pair,
#     we can 'jump' to the paired boundary cell and continue BFS from there.
#     """
#     if start == goal:
#         return [start]
    
#     queue = deque()
#     queue.append((start, [start]))
#     visited = {start}
    
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),           (0, 1),
#                   (1, -1),  (1, 0),  (1, 1)]
    
#     while queue:
#         current, path = queue.popleft()
        
#         # 1) Normal BFS expansions
#         for dx, dy in directions:
#             nxt = (current[0] + dx, current[1] + dy)
#             if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE:
#                 if nxt == goal:
#                     return path + [nxt]
#                 if nxt not in obstacles and nxt not in visited:
#                     visited.add(nxt)
#                     queue.append((nxt, path + [nxt]))
        
#         # 2) Jump BFS expansions (if allowed and current is a boundary cell with a portal)
#         if allow_jump and current in BOUNDARY_PORTALS:
#             jump_target = BOUNDARY_PORTALS[current]
#             if jump_target and jump_target not in visited:
#                 # We "teleport" to jump_target and continue BFS from there
#                 visited.add(jump_target)
#                 new_path = path + [jump_target]
#                 if jump_target == goal:
#                     return new_path
#                 queue.append((jump_target, new_path))
    
#     return None  # no path found

# ##############################################################################
# # 3. AGENT CLASS
# ##############################################################################

# class Agent:
#     def __init__(self, x, y, agent_id):
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id
#         self.target_block = None
#         self.boundary_flag = False  # set True if agent is part of boundary ring
    
#     def compute_next_move(self, occupied_positions, boundary_formed=False):
#         """
#         If boundary_formed, we allow BFS with jump.
#         Otherwise, we do a normal BFS.
#         """
#         if self.target_block and (self.x, self.y) != self.target_block:
#             # Remove self from obstacles to let BFS start from current cell
#             obstacles = occupied_positions.copy()
#             obstacles.discard((self.x, self.y))
            
#             path = None
#             if boundary_formed:
#                 # BFS with jump allowed
#                 path = bfs_with_jump(
#                     start=(self.x, self.y),
#                     goal=self.target_block,
#                     obstacles=obstacles,
#                     allow_jump=True
#                 )
#             else:
#                 # Normal BFS
#                 path = bfs_with_jump(
#                     start=(self.x, self.y),
#                     goal=self.target_block,
#                     obstacles=obstacles,
#                     allow_jump=False
#                 )
            
#             if path and len(path) >= 2:
#                 return path[1]  # next step
#         return (self.x, self.y)  # no move

# ##############################################################################
# # 4. CREATE AGENTS, DIAMOND, ETC. (SAME AS BEFORE)
# ##############################################################################

# def create_agents():
#     agent_positions = set()
#     agents = []
#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)
#         y = np.random.randint(0, GRID_SIZE)
#         if (x, y) not in agent_positions:
#             agent_positions.add((x, y))
#             agents.append(Agent(x, y, len(agents)))
#     return agents

# def generate_diamond():
#     center_x, center_y = 6, 5
#     diamond_positions = set()
#     offsets = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2),
#         (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1),
#         (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]
#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) != FORBIDDEN_DESTINATION:
#             diamond_positions.add((x, y))
#     sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
#     upper_half = set(sorted_blocks[:10])
#     lower_half = set(sorted_blocks[10:])
#     return upper_half, lower_half, diamond_positions

# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
#     (st.session_state.diamond_upper_half,
#      st.session_state.diamond_lower_half,
#      st.session_state.diamond_positions) = generate_diamond()

# def get_occupied_positions():
#     return {(ag.x, ag.y) for ag in st.session_state.agents}

# ##############################################################################
# # 5. DIVIDE & CONQUER DESTINATION ASSIGNMENT (UNCHANGED)
# ##############################################################################

# def assign_destinations():
#     occupied_destinations = set()
#     agents_sorted = sorted(st.session_state.agents, key=lambda a: -a.x)
#     for agent in agents_sorted:
#         if agent.target_block is not None:
#             continue
#         # Try upper half first
#         available_upper = st.session_state.diamond_upper_half - occupied_destinations
#         if available_upper:
#             pq = []
#             for block in available_upper:
#                 distance = abs(agent.x - block[0]) + abs(agent.y - block[1])
#                 heapq.heappush(pq, (-distance, block))
#             if pq:
#                 _, block = heapq.heappop(pq)
#                 agent.target_block = block
#                 occupied_destinations.add(block)
#                 continue
#         # Otherwise pick from lower half
#         available_lower = st.session_state.diamond_lower_half - occupied_destinations
#         if available_lower:
#             pq = []
#             for block in available_lower:
#                 distance = abs(agent.x - block[0]) + abs(agent.y - block[1])
#                 heapq.heappush(pq, (-distance, block))
#             if pq:
#                 _, block = heapq.heappop(pq)
#                 agent.target_block = block
#                 occupied_destinations.add(block)

# if "destinations_assigned" not in st.session_state:
#     assign_destinations()
#     st.session_state.destinations_assigned = True

# ##############################################################################
# # 6. CHECK IF BOUNDARY IS FORMED
# ##############################################################################

# def check_boundary_formed():
#     """
#     If all BOUNDARY_CELLS are occupied, we mark those occupant agents as boundary_flag=True
#     and return True. Otherwise False.
#     """
#     occupied_positions = get_occupied_positions()
#     if BOUNDARY_CELLS.issubset(occupied_positions):
#         for ag in st.session_state.agents:
#             if (ag.x, ag.y) in BOUNDARY_CELLS:
#                 ag.boundary_flag = True
#         return True
#     return False

# ##############################################################################
# # 7. PARALLEL MOVE WITH COLLISION AVOIDANCE & JUMP BFS
# ##############################################################################

# grid_placeholder = st.empty()

# def plot_grid():
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)
    
#     # Draw diamond
#     for x, y in st.session_state.diamond_positions:
#         ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))
    
#     # (Optional) highlight boundary cells
#     for (bx, by) in BOUNDARY_CELLS:
#         ax.add_patch(plt.Rectangle((by, bx), 1, 1, color="blue", alpha=0.2))

#     # Agents
#     for agent in st.session_state.agents:
#         color = "red"
#         if agent.boundary_flag:
#             color = "purple"  # boundary occupant
#         ax.scatter(agent.y + 0.5, agent.x + 0.5, color=color, s=800, marker="o", edgecolors="black")
#         ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white",
#                 ha='center', va='center', fontsize=12)
    
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)
#     grid_placeholder.pyplot(fig)

# def parallel_move():
#     occupied_positions = get_occupied_positions()
#     proposed_moves = {}
#     agent_moves = {}
    
#     # Check if boundary is formed
#     boundary_formed = check_boundary_formed()
    
#     # Each agent proposes a move
#     for agent in st.session_state.agents:
#         if (agent.x, agent.y) == agent.target_block:
#             agent_moves[agent.agent_id] = (agent.x, agent.y)
#         else:
#             next_move = agent.compute_next_move(occupied_positions, boundary_formed)
#             agent_moves[agent.agent_id] = next_move
#             proposed_moves.setdefault(next_move, []).append(agent)
    
#     # Collision resolution
#     allowed_moves = {}
#     for cell, agents_list in proposed_moves.items():
#         if len(agents_list) == 1:
#             allowed_moves[agents_list[0].agent_id] = cell
#         else:
#             # The agent whose target has the higher x gets priority
#             agents_list.sort(key=lambda a: (a.target_block[0], -a.agent_id), reverse=True)
#             winner = agents_list[0]
#             allowed_moves[winner.agent_id] = cell
#             # Others remain in place
#             for loser in agents_list[1:]:
#                 allowed_moves[loser.agent_id] = (loser.x, loser.y)
    
#     # Update positions
#     for agent in st.session_state.agents:
#         new_position = allowed_moves.get(agent.agent_id, (agent.x, agent.y))
#         agent.x, agent.y = new_position
    
#     plot_grid()

# ##############################################################################
# # 8. STREAMLIT
# ##############################################################################

# if st.button("Move Next Step"):
#     parallel_move()
#     time.sleep(0.3)

# plot_grid()
# st.write("**Agents move in parallel; if boundary is formed, BFS can jump across sides!**")












# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import heapq
# import time
# from collections import deque

# GRID_SIZE = 10
# AGENT_COUNT = 20
# FORBIDDEN_DESTINATION = (4, 3)

# ##############################################################################
# # 1. DEFINE SPECIAL CELLS (ADAPT THESE TO YOUR ACTUAL SCENARIO)
# ##############################################################################

# # Example boundary cells (16 total). You must fill these in to match your shape.
# BOUNDARY_CELLS = {
#     # (x, y) for the 16 boundary positions
#     # e.g. (5,4), (5,5), ...
#     # Just placeholders below:
#     (5,4), (5,5), (5,6), (5,7),
#     (6,3), (6,8),
#     (7,3), (7,8),
#     (8,4), (8,5), (8,6), (8,7),
#     (6,4), (6,5), (6,6), (6,7)
# }

# # The four "circled" positions from picture 2.
# CIRCLED_CELLS = {(6,4), (6,8), (8,4), (8,8)}

# # The "fallback" cells from picture 3 (occupants can also move deeper).
# FALLBACK_CELLS = {
#     # Put your actual fallback cells here.
#     (7,4), (7,5), (7,6), (7,7)
# }

# ##############################################################################
# # 2. BFS FUNCTION (UNCHANGED)
# ##############################################################################

# def bfs(start, goal, obstacles):
#     if start == goal:
#         return [start]
#     from collections import deque
#     queue = deque()
#     queue.append((start, [start]))
#     visited = {start}
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (0, -1),           (0, 1),
#                   (1, -1),  (1, 0),  (1, 1)]
#     while queue:
#         current, path = queue.popleft()
#         for dx, dy in directions:
#             next_cell = (current[0] + dx, current[1] + dy)
#             if 0 <= next_cell[0] < GRID_SIZE and 0 <= next_cell[1] < GRID_SIZE:
#                 if next_cell == goal:
#                     return path + [next_cell]
#                 if next_cell not in obstacles and next_cell not in visited:
#                     visited.add(next_cell)
#                     queue.append((next_cell, path + [next_cell]))
#     return None

# ##############################################################################
# # 3. AGENT CLASS WITH A BOUNDARY FLAG
# ##############################################################################

# class Agent:
#     def __init__(self, x, y, agent_id):
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id
#         self.target_block = None
#         # New: boundary_flag indicates agent is “locked” on the boundary.
#         self.boundary_flag = False

#     def compute_next_move(self, occupied_positions):
#         """Use BFS to get next step toward target_block, if possible."""
#         if self.target_block and (self.x, self.y) != self.target_block:
#             obstacles = occupied_positions.copy()
#             obstacles.discard((self.x, self.y))
#             path = bfs((self.x, self.y), self.target_block, obstacles)
#             if path and len(path) >= 2:
#                 return path[1]
#         return (self.x, self.y)

# ##############################################################################
# # 4. CREATE AGENTS AND DIAMOND (SAME AS BEFORE)
# ##############################################################################

# def create_agents():
#     agent_positions = set()
#     agents = []
#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)  # rows 0 or 1
#         y = np.random.randint(0, GRID_SIZE)
#         if (x, y) not in agent_positions:
#             agent_positions.add((x, y))
#             agents.append(Agent(x, y, len(agents)))
#     return agents

# def generate_diamond():
#     center_x, center_y = 6, 5
#     diamond_positions = set()
#     offsets = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2),
#         (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1),
#         (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]
#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) != FORBIDDEN_DESTINATION:
#             diamond_positions.add((x, y))
#     sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])
#     upper_half = set(sorted_blocks[:10])
#     lower_half = set(sorted_blocks[10:])
#     return upper_half, lower_half, diamond_positions

# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
#     (st.session_state.diamond_upper_half,
#      st.session_state.diamond_lower_half,
#      st.session_state.diamond_positions) = generate_diamond()

# ##############################################################################
# # 5. DESTINATION ASSIGNMENT (AS BEFORE) — DIVIDE & CONQUER
# ##############################################################################

# def get_occupied_positions():
#     return {(agent.x, agent.y) for agent in st.session_state.agents}

# def assign_destinations():
#     occupied_destinations = set()
#     agents_sorted = sorted(st.session_state.agents, key=lambda a: -a.x)
#     for agent in agents_sorted:
#         if agent.target_block is not None:
#             continue
#         available_upper = st.session_state.diamond_upper_half - occupied_destinations
#         if available_upper:
#             pq = []
#             for block in available_upper:
#                 distance = abs(agent.x - block[0]) + abs(agent.y - block[1])
#                 heapq.heappush(pq, (-distance, block))
#             if pq:
#                 _, block = heapq.heappop(pq)
#                 agent.target_block = block
#                 occupied_destinations.add(block)
#                 continue
#         # Otherwise pick from lower half
#         available_lower = st.session_state.diamond_lower_half - occupied_destinations
#         if available_lower:
#             pq = []
#             for block in available_lower:
#                 distance = abs(agent.x - block[0]) + abs(agent.y - block[1])
#                 heapq.heappush(pq, (-distance, block))
#             if pq:
#                 _, block = heapq.heappop(pq)
#                 agent.target_block = block
#                 occupied_destinations.add(block)

# if "destinations_assigned" not in st.session_state:
#     assign_destinations()
#     st.session_state.destinations_assigned = True

# ##############################################################################
# # 6. CHECK IF BOUNDARY IS FORMED
# ##############################################################################

# def check_boundary_formed():
#     """
#     If all boundary cells are occupied, set boundary_formed = True
#     and mark the agents occupying those cells as boundary_flag = True.
#     """
#     occupied_positions = get_occupied_positions()
#     if BOUNDARY_CELLS.issubset(occupied_positions):
#         # Mark those occupant agents
#         for agent in st.session_state.agents:
#             if (agent.x, agent.y) in BOUNDARY_CELLS:
#                 agent.boundary_flag = True
#         return True
#     return False

# ##############################################################################
# # 7. TRY TO REASSIGN OUTSIDE AGENTS THAT ARE BLOCKED
# ##############################################################################

# def can_agent_move_deeper(agent, occupied_positions):
#     """
#     Check if 'agent' can move from its current cell to any deeper cell
#     inside the shape (or any free cell you define as "deeper").
#     This is a placeholder function—you must define your own logic/criteria
#     for what counts as a "deeper" cell.
#     """
#     # Example: We assume any free cell inside st.session_state.diamond_positions
#     # that is not BOUNDARY_CELLS is considered deeper.
#     deeper_cells = st.session_state.diamond_positions - BOUNDARY_CELLS
#     # Remove agent's own position from obstacles
#     obstacles = occupied_positions.copy()
#     obstacles.discard((agent.x, agent.y))
#     # Try BFS from agent’s current position to any free deeper cell.
#     for cell in deeper_cells:
#         if cell not in obstacles:
#             path = bfs((agent.x, agent.y), cell, obstacles)
#             if path:
#                 return cell  # Return the first deeper cell we can reach
#     return None

# def reassign_blocked_agents():
#     """
#     If boundary_formed == True, any agent outside that cannot BFS to its
#     original target tries to get a new destination among:
#       1) CIRCLED_CELLS (if occupant can move deeper)
#       2) FALLBACK_CELLS (if occupant can move deeper)
#     """
#     occupied_positions = get_occupied_positions()
#     boundary_formed = check_boundary_formed()
#     if not boundary_formed:
#         return  # Do nothing if boundary is not fully formed

#     # For each agent that has NOT reached its target, check if BFS is still possible.
#     for agent in st.session_state.agents:
#         if (agent.x, agent.y) == agent.target_block:
#             continue  # Already at destination
#         # If BFS is possible, no need to reassign.
#         obstacles = occupied_positions.copy()
#         obstacles.discard((agent.x, agent.y))
#         path_to_original = bfs((agent.x, agent.y), agent.target_block, obstacles)
#         if path_to_original:
#             continue  # We can still reach the original target

#         # BFS not possible => we attempt to reassign
#         # 1) Try circled cells
#         new_dest = try_reassign_to_special_cells(agent, CIRCLED_CELLS)
#         if not new_dest:
#             # 2) Try fallback cells
#             new_dest = try_reassign_to_special_cells(agent, FALLBACK_CELLS)
#         # If still None, agent remains blocked
#         if new_dest:
#             agent.target_block = new_dest

# def try_reassign_to_special_cells(outside_agent, candidate_cells):
#     """
#     Attempt to reassign 'outside_agent' to one of the 'candidate_cells'.
#     We only succeed if the occupant of that cell can move deeper.
#     Then we effectively 'swap':
#       - occupant moves to a deeper cell,
#       - outside_agent claims occupant's old cell as the new target.
#     Returns the newly assigned cell, or None if no swap could be done.
#     """
#     occupied_positions = get_occupied_positions()
#     # Sort candidate cells by closeness to outside_agent, or any heuristic you like
#     candidate_list = sorted(candidate_cells, 
#                             key=lambda c: abs(c[0]-outside_agent.x) + abs(c[1]-outside_agent.y))
#     for cell in candidate_list:
#         # Is this cell currently occupied by someone?
#         occupant = None
#         for ag in st.session_state.agents:
#             if (ag.x, ag.y) == cell:
#                 occupant = ag
#                 break
#         if occupant is None:
#             # If it's unoccupied, we can just pick it
#             # (Though your logic suggests these special cells are occupied.)
#             return cell

#         # If occupant is found, check if occupant can move deeper
#         deeper_cell = can_agent_move_deeper(occupant, occupied_positions)
#         if deeper_cell:
#             # We do a "swap":
#             # occupant changes its target_block to deeper_cell
#             occupant.target_block = deeper_cell
#             # outside_agent can now claim occupant's cell as new target
#             return cell
#     return None

# ##############################################################################
# # 8. PLOTTING AND PARALLEL MOVEMENT
# ##############################################################################

# grid_placeholder = st.empty()

# def plot_grid():
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)

#     # Draw the diamond shape
#     for x, y in st.session_state.diamond_positions:
#         ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))

#     # Optional: color boundary cells differently
#     for (bx, by) in BOUNDARY_CELLS:
#         ax.add_patch(plt.Rectangle((by, bx), 1, 1, color="blue", alpha=0.2))

#     # Optional: highlight circled or fallback cells
#     for (cx, cy) in CIRCLED_CELLS:
#         ax.add_patch(plt.Rectangle((cy, cx), 1, 1, color="green", alpha=0.2))
#     for (fx, fy) in FALLBACK_CELLS:
#         ax.add_patch(plt.Rectangle((fy, fx), 1, 1, color="orange", alpha=0.2))

#     # Draw agents
#     for agent in st.session_state.agents:
#         color = "red"
#         if agent.boundary_flag:
#             color = "purple"  # or something to mark boundary agents
#         ax.scatter(agent.y + 0.5, agent.x + 0.5, color=color, s=800, marker="o", edgecolors="black")
#         ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white",
#                 ha='center', va='center', fontsize=12)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)
#     grid_placeholder.pyplot(fig)

# def parallel_move():
#     """
#     Standard parallel move with collision avoidance by target’s x-coordinate.
#     Then attempt reassignments for blocked agents if boundary is formed.
#     """
#     occupied_positions = get_occupied_positions()
#     proposed_moves = {}
#     agent_moves = {}

#     for agent in st.session_state.agents:
#         if (agent.x, agent.y) == agent.target_block:
#             agent_moves[agent.agent_id] = (agent.x, agent.y)
#         else:
#             next_move = agent.compute_next_move(occupied_positions)
#             agent_moves[agent.agent_id] = next_move
#             proposed_moves.setdefault(next_move, []).append(agent)

#     # Collision resolution
#     allowed_moves = {}
#     for cell, agents_list in proposed_moves.items():
#         if len(agents_list) == 1:
#             allowed_moves[agents_list[0].agent_id] = cell
#         else:
#             # The agent whose target has the higher x gets priority
#             agents_list.sort(key=lambda a: (a.target_block[0], -a.agent_id), reverse=True)
#             winner = agents_list[0]
#             allowed_moves[winner.agent_id] = cell
#             for loser in agents_list[1:]:
#                 allowed_moves[loser.agent_id] = (loser.x, loser.y)

#     # Update positions
#     for agent in st.session_state.agents:
#         new_position = allowed_moves.get(agent.agent_id, (agent.x, agent.y))
#         agent.x, agent.y = new_position

#     # Now check boundary formation and try dynamic reassignments
#     reassign_blocked_agents()

#     plot_grid()

# ##############################################################################
# # 9. STREAMLIT BUTTON
# ##############################################################################

# if st.button("Move Next Step"):
#     parallel_move()
#     time.sleep(0.3)

# plot_grid()
# st.write("**Agents move in parallel, forming a boundary, and reassigning blocked agents!**")
