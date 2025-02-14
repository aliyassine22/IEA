import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Constants
GRID_SIZE = 10  # The grid is a 10x10 square
AGENT_COUNT = 20  # Number of agents

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
        """Calculate Manhattan distance to each allowed block and store in a priority queue (max heap)."""
        pq = []
        distance_map = {}

        for (dx, dy) in allowed_blocks:
            distance = abs(self.x - dx) + abs(self.y - dy)  # Manhattan distance
            heapq.heappush(pq, (-distance, (dx, dy)))  # Store negative distance (max heap behavior)
            distance_map[-distance] = (dx, dy)

        return pq, distance_map

    def assign_destination(self, occupied_positions, allowed_blocks):
        """Assign the farthest available block within the agent's designated half."""
        pq, distance_map = self.calculate_manhattan_distances(allowed_blocks)

        while pq:
            farthest_distance, destination = heapq.heappop(pq)  # Get farthest block (-distance)
            if destination not in occupied_positions:
                self.target_block = destination
                occupied_positions.add(destination)
                return

# ----------------------- CREATE AGENTS FUNCTION -----------------------
def create_agents():
    """Initialize 20 agents in the bottom two rows with unique positions."""
    agent_positions = set()
    agents = []

    while len(agents) < AGENT_COUNT:
        x = np.random.randint(0, 2)  # Ensure agents are only in rows 0 and 1
        y = np.random.randint(0, GRID_SIZE)

        if (x, y) not in agent_positions:
            agent_positions.add((x, y))
            agents.append(Agent(x, y, len(agents)))

    return agents

# ----------------------- GENERATE 20-BLOCK DIAMOND FUNCTION -----------------------
def generate_diamond():
    """Generate a diamond shape with exactly 20 blocks, avoiding the bottom 2 rows."""
    center_x, center_y = 6, 5  # Centered in the grid
    diamond_positions = set()

    # Diamond Structure (20 Blocks)
    offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, -1), (-2, 1), (2, -1), (2, 1),
        (-1, -2), (-1, 2), (1, -2), (1, 2)
    ]

    for dx, dy in offsets:
        x, y = center_x + dx, center_y + dy
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            diamond_positions.add((x, y))

    # Remove the center block (6,5) to keep exactly 20 blocks
    diamond_positions.discard((center_x, center_y))

    # Split the diamond into upper and lower halves
    sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])  # Sort by x descending
    upper_half = set(sorted_blocks[:10])  # Top 10 blocks
    lower_half = set(sorted_blocks[10:])  # Bottom 10 blocks

    return upper_half, lower_half

# ----------------------- STREAMLIT SESSION INITIALIZATION -----------------------
if "agents" not in st.session_state:
    st.session_state.agents = create_agents()  # Create agents

if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
    st.session_state.diamond_upper_half, st.session_state.diamond_lower_half = generate_diamond()  # Create diamond

# ----------------------- GET OCCUPIED POSITIONS FUNCTION -----------------------
def get_occupied_positions():
    """Returns a set of positions occupied by agents."""
    return {(agent.x, agent.y) for agent in st.session_state.agents}

# ----------------------- ASSIGN DESTINATION BLOCKS TO AGENTS -----------------------
occupied_positions = get_occupied_positions()

for agent in st.session_state.agents:
    if agent.x == 1:  # Higher row agents
        agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half)
    else:  # Lower row agents
        agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half)

# ----------------------- GRID VISUALIZATION FUNCTION -----------------------
def plot_grid(agents, grid_placeholder):
    """Plots the grid and updates the placeholder in Streamlit for live animation."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)

    # Plot the diamond shape (gray squares)
    for x, y in st.session_state.diamond_upper_half | st.session_state.diamond_lower_half:
        ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))

    # Plot agents at their original positions
    for agent in agents:
        ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
        ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)

        # Mark the selected target block
        if agent.target_block:
            target_x, target_y = agent.target_block
            ax.add_patch(plt.Rectangle((target_y, target_x), 1, 1, color="blue", alpha=0.5))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    grid_placeholder.pyplot(fig)

# ----------------------- DISPLAY INITIAL GRID -----------------------
grid_placeholder = st.empty()
plot_grid(st.session_state.agents, grid_placeholder)

# ----------------------- STREAMLIT BUTTON TO ASSIGN DESTINATIONS -----------------------
if st.button("Assign Destinations"):
    occupied_positions = get_occupied_positions()
    for agent in st.session_state.agents:
        if agent.x == 1:
            agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half)
        else:
            agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half)

    plot_grid(st.session_state.agents, grid_placeholder)

    # Print agent destinations
    st.write("### **Agent Destinations:**")
    for agent in st.session_state.agents:
        if agent.target_block:
            st.write(f"Agent {agent.agent_id}: Start ({agent.x}, {agent.y}) → Destination {agent.target_block}")
        else:
            st.write(f"Agent {agent.agent_id}: No assigned destination")

st.write("**Agents have chosen their respective destinations!**")

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import heapq

# # Constants
# GRID_SIZE = 10
# AGENT_COUNT = 20

# # Define the Agent class
# class Agent:
#     def __init__(self, x, y, agent_id):
#         """Initialize agent with position (x, y) and unique ID."""
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id  # Unique identifier for the agent
#         self.target_block = None  # The assigned target block for this agent

#     def calculate_manhattan_distances(self, allowed_blocks):
#         """Calculate Manhattan distance to each allowed block and store in a priority queue (max heap)."""
#         pq = []
#         distance_map = {}

#         for (dx, dy) in allowed_blocks:
#             distance = abs(self.x - dx) + abs(self.y - dy)
#             heapq.heappush(pq, (-distance, (dx, dy)))  # Store negative distance for max heap
#             distance_map[-distance] = (dx, dy)

#         return pq, distance_map

#     def assign_destination(self, occupied_positions, allowed_blocks):
#         """Assign the farthest available block within the agent's designated half."""
#         pq, distance_map = self.calculate_manhattan_distances(allowed_blocks)

#         while pq:
#             farthest_distance, destination = heapq.heappop(pq)
#             if destination not in occupied_positions:
#                 self.target_block = destination
#                 occupied_positions.add(destination)
#                 return

#     def move_towards_target(self, occupied_positions):
#         """Move step-by-step toward the assigned block using Manhattan distance."""
#         if self.target_block:
#             target_x, target_y = self.target_block
#             dx = np.sign(target_x - self.x)
#             dy = np.sign(target_y - self.y)

#             if (self.x + dx, self.y) not in occupied_positions:
#                 self.x += dx
#             elif (self.x, self.y + dy) not in occupied_positions:
#                 self.y += dy

# # Function to create agents in bottom two rows
# def create_agents():
#     """Initialize 20 agents in the bottom two rows with unique positions."""
#     agent_positions = set()
#     agents = []

#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)  # Ensure agents are only in rows 0 and 1
#         y = np.random.randint(0, GRID_SIZE)

#         if (x, y) not in agent_positions:
#             agent_positions.add((x, y))
#             agents.append(Agent(x, y, len(agents)))

#     return agents

# # Function to generate a diamond shape with exactly 20 blocks
# def generate_diamond():
#     """Generate a diamond shape with exactly 20 blocks, avoiding the bottom 2 rows."""
#     center_x, center_y = 6, 5
#     diamond_positions = set()

#     # Diamond Structure (20 Blocks) - Removed center block
#     offsets = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2),
#         (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1),
#         (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]

#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
#             diamond_positions.add((x, y))

#     # Remove the center block (6,5) to keep exactly 20 blocks
#     diamond_positions.discard((center_x, center_y))

#     # Split the diamond into upper and lower halves
#     sorted_blocks = sorted(diamond_positions, key=lambda pos: -pos[0])  # Sort by x descending
#     upper_half = set(sorted_blocks[:10])  # Top 10 blocks
#     lower_half = set(sorted_blocks[10:])  # Bottom 10 blocks

#     return upper_half, lower_half

# # Initialize Streamlit session state
# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
#     st.session_state.diamond_upper_half, st.session_state.diamond_lower_half = generate_diamond()

# # Function to get occupied positions
# def get_occupied_positions():
#     """Returns a set of positions occupied by agents."""
#     return {(agent.x, agent.y) for agent in st.session_state.agents}

# # Assign target blocks to agents based on their row
# occupied_positions = get_occupied_positions()
# for agent in st.session_state.agents:
#     if agent.x == 1:  # Higher row agents
#         agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half)
#     else:  # Lower row agents
#         agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half)

# # Move agents towards their destinations
# for agent in st.session_state.agents:
#     agent.move_towards_target(occupied_positions)

# # Function to display the grid
# def plot_grid(agents, grid_placeholder):
#     """Plots the grid and updates the placeholder in Streamlit for live animation."""
#     fig, ax = plt.subplots(figsize=(6, 6))

#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)

#     # Draw grid lines
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)

#     # Plot the diamond shape (gray squares)
#     for x, y in st.session_state.diamond_upper_half | st.session_state.diamond_lower_half:
#         ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))

#     # Plot agents
#     for agent in agents:
#         ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
#         ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)

#     grid_placeholder.pyplot(fig)

# # Create a placeholder for the grid (for real-time updates)
# grid_placeholder = st.empty()
# plot_grid(st.session_state.agents, grid_placeholder)

# # Button to trigger movement
# if st.button("Move Agents"):
#     for agent in st.session_state.agents:
#         agent.move_towards_target(get_occupied_positions())

#     plot_grid(st.session_state.agents, grid_placeholder)

# st.write("**Agents now correctly target only their respective halves of the diamond!**")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import heapq

# # Constants
# GRID_SIZE = 10
# AGENT_COUNT = 20

# # Define the Agent class
# class Agent:
#     def __init__(self, x, y, agent_id):
#         """Initialize agent with position (x, y) and unique ID."""
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id  # Unique identifier for the agent
#         self.target_block = None  # The assigned target block for this agent

#     def calculate_manhattan_distances(self, diamond_half):
#         """Calculate Manhattan distance to each allowed diamond block and store in a priority queue (max heap)."""
#         pq = []
#         distance_map = {}  # {distance: (x, y)}

#         for (dx, dy) in diamond_half:
#             distance = abs(self.x - dx) + abs(self.y - dy)
#             heapq.heappush(pq, (-distance, (dx, dy)))  # Store negative distance for max heap
#             distance_map[-distance] = (dx, dy)

#         return pq, distance_map

#     def assign_destination(self, occupied_positions, diamond_half):
#         """Assign the farthest available diamond block within the agent's designated half."""
#         pq, distance_map = self.calculate_manhattan_distances(diamond_half)

#         while pq:
#             farthest_distance, destination = heapq.heappop(pq)  # Get the farthest block (-distance)
#             if destination not in occupied_positions:  # Ensure block is available
#                 self.target_block = destination  # Assign block
#                 occupied_positions.add(destination)  # Reserve block
#                 return

#     def move_towards_target(self, occupied_positions):
#         """Move step-by-step toward the assigned block using Manhattan distance."""
#         if self.target_block:
#             target_x, target_y = self.target_block
#             dx = np.sign(target_x - self.x)  # Determine movement direction (x-axis)
#             dy = np.sign(target_y - self.y)  # Determine movement direction (y-axis)

#             if (self.x + dx, self.y) not in occupied_positions:
#                 self.x += dx  # Move in x-direction if possible
#             elif (self.x, self.y + dy) not in occupied_positions:
#                 self.y += dy  # Move in y-direction if possible

# # Function to create agents
# def create_agents():
#     """Initialize 20 agents in the bottom two rows with unique positions."""
#     agent_positions = set()
#     agents = []

#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)  # Bottom two rows (0 or 1)
#         y = np.random.randint(0, GRID_SIZE)

#         if (x, y) not in agent_positions:  # Avoid duplicate positions
#             agent_positions.add((x, y))
#             agents.append(Agent(x, y, len(agents)))  # Create an Agent object with ID

#     return agents  # Return a list of Agent objects

# # Function to generate a diamond shape with exactly 20 blocks
# def generate_diamond():
#     """Generate a diamond shape with exactly 20 blocks, avoiding the bottom 2 rows."""
#     center_x, center_y = 6, 5  # Centered in the grid (avoiding bottom 2 rows)
#     diamond_positions = set()
    
#     # Diamond Structure (20 Blocks) - Corrected
#     offsets = [
#         (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2), 
#         (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1), 
#         (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]  # Removed (0, 0) to maintain exactly 20 blocks

#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
#             diamond_positions.add((x, y))

#     # Split diamond into upper and lower halves
#     upper_half = {pos for pos in diamond_positions if pos[0] >= center_x}
#     lower_half = {pos for pos in diamond_positions if pos[0] < center_x}

#     return upper_half, lower_half

# # Initialize Streamlit session state
# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# if "diamond_upper_half" not in st.session_state or "diamond_lower_half" not in st.session_state:
#     st.session_state.diamond_upper_half, st.session_state.diamond_lower_half = generate_diamond()

# # Function to get occupied positions
# def get_occupied_positions():
#     """Returns a set of positions occupied by agents."""
#     return {(agent.x, agent.y) for agent in st.session_state.agents}

# # Assign target blocks to agents based on their row
# occupied_positions = get_occupied_positions()
# for agent in st.session_state.agents:
#     if agent.x == 1:  # Higher row agents
#         agent.assign_destination(occupied_positions, st.session_state.diamond_upper_half)
#     else:  # Lower row agents
#         agent.assign_destination(occupied_positions, st.session_state.diamond_lower_half)

# # Move agents towards their destinations
# for agent in st.session_state.agents:
#     agent.move_towards_target(occupied_positions)

# # Function to display the grid
# def plot_grid(agents, grid_placeholder):
#     """Plots the grid and updates the placeholder in Streamlit for live animation."""
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Setting up grid limits
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)

#     # Draw grid lines
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)

#     # Plot the diamond shape (gray squares)
#     for x, y in st.session_state.diamond_upper_half | st.session_state.diamond_lower_half:
#         ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))  # Mark the diamond shape

#     # Plot agents
#     for agent in agents:
#         ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")
#         ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)

#     # Remove axis labels for a clean look
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)

#     grid_placeholder.pyplot(fig)  # Update the Streamlit container with new figure

# # Create a placeholder for the grid (for real-time updates)
# grid_placeholder = st.empty()
# plot_grid(st.session_state.agents, grid_placeholder)

# # Button to trigger movement
# if st.button("Move Agents"):
#     for agent in st.session_state.agents:
#         agent.move_towards_target(get_occupied_positions())

#     plot_grid(st.session_state.agents, grid_placeholder)

# st.write("**Agents are now correctly targeting their respective halves of the diamond!**")



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# GRID_SIZE = 10
# AGENT_COUNT = 20

# # Define the Agent class
# class Agent:
#     def __init__(self, x, y, agent_id):
#         """Initialize agent with position (x, y) and unique ID."""
#         self.x = x
#         self.y = y
#         self.agent_id = agent_id  # Unique identifier for the agent

#     def move(self, dx, dy, occupied_positions):
#         """Move the agent by (dx, dy), ensuring it stays within the grid and avoids occupied spaces."""
#         new_x = max(0, min(GRID_SIZE - 1, self.x + dx))
#         new_y = max(0, min(GRID_SIZE - 1, self.y + dy))

#         # Ensure the new position is not occupied (agents or diamond)
#         if (new_x, new_y) not in occupied_positions and (new_x, new_y) not in diamond_positions:
#             self.x, self.y = new_x, new_y  # Update position

#     # Individual movement functions with constraints
#     def move_up(self, occupied_positions):
#         if self.x < GRID_SIZE - 1:
#             self.move(1, 0, occupied_positions)

#     def move_down(self, occupied_positions):
#         if self.x > 0:
#             self.move(-1, 0, occupied_positions)

#     def move_left(self, occupied_positions):
#         if self.y > 0:
#             self.move(0, -1, occupied_positions)

#     def move_right(self, occupied_positions):
#         if self.y < GRID_SIZE - 1:
#             self.move(0, 1, occupied_positions)

#     def move_up_right(self, occupied_positions):
#         if self.x < GRID_SIZE - 1 and self.y < GRID_SIZE - 1:
#             self.move(1, 1, occupied_positions)

#     def move_up_left(self, occupied_positions):
#         if self.x < GRID_SIZE - 1 and self.y > 0:
#             self.move(1, -1, occupied_positions)

#     def move_down_right(self, occupied_positions):
#         if self.x > 0 and self.y < GRID_SIZE - 1:
#             self.move(-1, 1, occupied_positions)

#     def move_down_left(self, occupied_positions):
#         if self.x > 0 and self.y > 0:
#             self.move(-1, -1, occupied_positions)

# # Function to create agents
# def create_agents():
#     """Initialize 20 agents in the bottom two rows with unique positions."""
#     agent_positions = set()
#     agents = []

#     while len(agents) < AGENT_COUNT:
#         x = np.random.randint(0, 2)  # Bottom two rows (0 or 1)
#         y = np.random.randint(0, GRID_SIZE)

#         if (x, y) not in agent_positions:  # Avoid duplicate positions
#             agent_positions.add((x, y))
#             agents.append(Agent(x, y, len(agents)))  # Create an Agent object with ID

#     return agents  # Return a list of Agent objects

# # Function to generate a diamond shape with 20 blocks
# def generate_diamond():
#     """Generate a diamond shape with exactly 20 blocks, avoiding the bottom 2 rows."""
#     center_x, center_y = 6, 5  # Centered in the grid (avoiding bottom 2 rows)
#     diamond_positions = set()
    
#     # Diamond Structure (20 Blocks)
#     offsets = [
#         (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
#         (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1),
#         (-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)
#     ]

#     for dx, dy in offsets:
#         x, y = center_x + dx, center_y + dy
#         if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
#             diamond_positions.add((x, y))

#     return diamond_positions

# # Store diamond shape positions
# diamond_positions = generate_diamond()

# # Function to get occupied positions
# def get_occupied_positions(exclude_agent=None):
#     """Returns a set of positions occupied by agents, excluding a specified agent."""
#     return {(agent.x, agent.y) for agent in st.session_state.agents if agent != exclude_agent}

# # Streamlit UI
# st.title("10x10 Grid with Diamond and Movement Constraints")

# # Initialize agents only once (to keep state)
# if "agents" not in st.session_state:
#     st.session_state.agents = create_agents()

# # Initialize selected agent index if not set
# if "selected_agent_idx" not in st.session_state:
#     st.session_state.selected_agent_idx = 0

# # Get the currently selected agent
# selected_agent = st.session_state.agents[st.session_state.selected_agent_idx]

# # Function to display the grid
# def plot_grid(agents, selected_agent, grid_placeholder):
#     """Plots the grid and updates the placeholder in Streamlit for live animation."""
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Setting up grid limits
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)

#     # Draw grid lines
#     for i in range(GRID_SIZE + 1):
#         ax.axhline(i, color="black", linewidth=0.5)
#         ax.axvline(i, color="black", linewidth=0.5)

#     # Plot the diamond shape (gray squares)
#     for x, y in diamond_positions:
#         ax.add_patch(plt.Rectangle((y, x), 1, 1, color="gray", alpha=0.5))  # Mark the diamond shape

#     # Plot agents with highlighting for selected agent
#     for agent in agents:
#         if agent == selected_agent:
#             ax.scatter(agent.y + 0.5, agent.x + 0.5, color="blue", s=1000, marker="o", edgecolors="yellow", linewidth=2)  # Highlight selected agent
#         else:
#             ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")

#         ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)

#     # Remove axis labels for a clean look
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_frame_on(False)

#     grid_placeholder.pyplot(fig)  # Update the Streamlit container with new figure

# # Create a placeholder for the grid (for real-time updates)
# grid_placeholder = st.empty()
# plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

# # Show the currently selected agent
# st.write(f"**Selected Agent ID:** {selected_agent.agent_id}")

# # Button to switch agents
# if st.button("Switch to Next Agent"):
#     st.session_state.selected_agent_idx = (st.session_state.selected_agent_idx + 1) % AGENT_COUNT
#     selected_agent = st.session_state.agents[st.session_state.selected_agent_idx]
#     plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

# # Store the diamond positions as goal states
# st.session_state.goal_positions = diamond_positions

# st.write("**Diamond shape has been set up! Agents cannot move through it.**")
