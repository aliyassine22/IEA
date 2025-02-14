import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 10
AGENT_COUNT = 20

# Define the Agent class
class Agent:
    def __init__(self, x, y, agent_id):
        """Initialize agent with position (x, y) and unique ID."""
        self.x = x
        self.y = y
        self.agent_id = agent_id  # Unique identifier for the agent

    def move(self, dx, dy, occupied_positions):
        """Move the agent by (dx, dy), ensuring it stays within the grid and avoids occupied spaces."""
        new_x = max(0, min(GRID_SIZE - 1, self.x + dx))
        new_y = max(0, min(GRID_SIZE - 1, self.y + dy))

        if (new_x, new_y) not in occupied_positions:  # Check if space is free
            self.x, self.y = new_x, new_y  # Update position

    # Individual movement functions
    def move_up(self, occupied_positions): self.move(1, 0, occupied_positions)
    def move_down(self, occupied_positions): self.move(-1, 0, occupied_positions)
    def move_left(self, occupied_positions): self.move(0, -1, occupied_positions)
    def move_right(self, occupied_positions): self.move(0, 1, occupied_positions)
    def move_up_right(self, occupied_positions): self.move(1, 1, occupied_positions)
    def move_up_left(self, occupied_positions): self.move(1, -1, occupied_positions)
    def move_down_right(self, occupied_positions): self.move(-1, 1, occupied_positions)
    def move_down_left(self, occupied_positions): self.move(-1, -1, occupied_positions)

# Function to create agents
def create_agents():
    """Initialize 20 agents in the bottom two rows with unique positions."""
    agent_positions = set()
    agents = []

    while len(agents) < AGENT_COUNT:
        x = np.random.randint(0, 2)  # Bottom two rows (0 or 1)
        y = np.random.randint(0, GRID_SIZE)
        
        if (x, y) not in agent_positions:  # Avoid duplicate positions
            agent_positions.add((x, y))
            agents.append(Agent(x, y, len(agents)))  # Create an Agent object with ID

    return agents  # Return a list of Agent objects

# Function to display the grid with circular agents
def plot_grid(agents, selected_agent, grid_placeholder):
    """Plots the grid and updates the placeholder in Streamlit for live animation."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Setting up grid limits
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i, color="black", linewidth=0.5)
        ax.axvline(i, color="black", linewidth=0.5)

    # Plot agents with highlighting for selected agent
    for agent in agents:
        if agent == selected_agent:
            ax.scatter(agent.y + 0.5, agent.x + 0.5, color="blue", s=1000, marker="o", edgecolors="yellow", linewidth=2)  # Highlight selected agent
        else:
            ax.scatter(agent.y + 0.5, agent.x + 0.5, color="red", s=800, marker="o", edgecolors="black")

        ax.text(agent.y + 0.5, agent.x + 0.5, str(agent.agent_id), color="white", ha='center', va='center', fontsize=12)

    # Remove axis labels for a clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    grid_placeholder.pyplot(fig)  # Update the Streamlit container with new figure

# Streamlit UI
st.title("10x10 Grid with Highlighted Selected Agent")

# Initialize agents only once (to keep state)
if "agents" not in st.session_state:
    st.session_state.agents = create_agents()

# Initialize selected agent index if not set
if "selected_agent_idx" not in st.session_state:
    st.session_state.selected_agent_idx = 0

# Function to get occupied positions
def get_occupied_positions(exclude_agent=None):
    """Returns a set of positions occupied by agents, excluding a specified agent."""
    return {(agent.x, agent.y) for agent in st.session_state.agents if agent != exclude_agent}

# Get the currently selected agent
selected_agent = st.session_state.agents[st.session_state.selected_agent_idx]

# Create a placeholder for the grid (for real-time updates)
grid_placeholder = st.empty()
plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

# Show the currently selected agent
st.write(f"**Selected Agent ID:** {selected_agent.agent_id}")

# Button to switch agents
if st.button("Switch to Next Agent"):
    st.session_state.selected_agent_idx = (st.session_state.selected_agent_idx + 1) % AGENT_COUNT
    selected_agent = st.session_state.agents[st.session_state.selected_agent_idx]
    plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

# Movement buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Move Up"):
        selected_agent.move_up(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

    if st.button("Move Left"):
        selected_agent.move_left(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

    if st.button("Move Up Left"):
        selected_agent.move_up_left(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

with col2:
    if st.button("Move Down"):
        selected_agent.move_down(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

    if st.button("Move Right"):
        selected_agent.move_right(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

with col3:
    if st.button("Move Up Right"):
        selected_agent.move_up_right(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

    if st.button("Move Down Right"):
        selected_agent.move_down_right(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)

    if st.button("Move Down Left"):
        selected_agent.move_down_left(get_occupied_positions(selected_agent))
        plot_grid(st.session_state.agents, selected_agent, grid_placeholder)
