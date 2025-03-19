# Multi-Agent Shape Formation Project

This project demonstrates three distinct implementations to form shapes using 20 agents on a 10x10 grid. Each implementation uses a different approach for goal assignment and collision-free movement, leveraging classic search algorithms and custom booking techniques.

---

## Overview

The project features three main implementations:

1. **Greedy Weight Map Algorithm**  
2. **Hungarian Booking + A\* Algorithm**  
3. **Custom Lane-Based Booking + A\* Algorithm**

Each method assigns destinations to agents and computes collision-free paths using variations on search algorithms and booking strategies.

---

## Implementation Details

### 1. Greedy Weight Map Algorithm

- **Graph Rewards (computeWeights):**  
  - **Destination Cells:**  
    The weight is higher the closer the cell is to the top (i.e. lower y value).  
  - **Non-Destination Cells:**  
    Weight is computed based on the distance (via BFS) to the nearest diamond cell.

- **Decision-Making:**  
  - **Row Priority:**  
    Decision-making occurs first in the top row and then in the bottom row.
  - **Within a Row:**  
    Decisions are made left-to-right, or alternatively, from the center outward.

- **Movement Logic (TakeMoveMoore):**  
  - First, all possible movements are evaluated.  
  - The move with the highest reward is selected.  
  - **Example:**  
    - Staying in Place: Reward = 2  
    - Moving Top Right: Reward = 4  
    - Moving Top: Reward = 3  
  - **Tie-Breaker:**  
    If two cells yield the same reward, the agent moves toward the cell closest to an unfilled position in the desired shape.

- **Running the Implementation:**  
  Use Streamlit to run the UI:
  ```bash
  streamlit run greedyWeightMapUI.py
### 2. Hungarian Booking + A* Algorithm

- **Booking Phase:**  
  - Uses the Hungarian algorithm to assign each agent a destination from the target shape.  
  - The cost is based on the Manhattan distance between an agent and a destination.

- **Path Planning Phase:**  
  - Employs an A* search to compute a collision-free path from the agent’s starting position to its assigned destination.  
  - A custom reservation table prevents agents from intersecting paths.

- **Running the Implementation:**  
  Execute the UI with:  
  ```bash
  python testUI.py
### 3. Custom Lane-Based Booking + A* Algorithm

- **Lane-Based Assignment:**  
  - Agents are split into two lists (upper and lower) based on their initial row.  
  - A lane map is created where keys are x-coordinates and values are heaps of available y-values (using negative values for max-heap behavior).  
  - Agents first try to get assigned a destination in their own column.  
  - If not available, they are assigned to the closest available lane based on their x-coordinate.

- **Path Planning:**  
  - An A* search (with a reservation table keyed by (position, timestep)) computes the collision-free path to the destination.  
  - The destination remains reserved for all future timesteps once reached.

- **Running the Implementation:**  
  Use Streamlit to run the UI for this algorithm:  
  ```bash
  streamlit run newAlgoUIcopy.py
 - **Technologies Used**
- Programming Language: Python 3
- Libraries & Tools:
- NumPy
- SciPy (Hungarian algorithm via linear_sum_assignment)
- Matplotlib (for plotting)
- Streamlit (for interactive UI)
- **Observations & Limitations**
- Greedy Weight Map:

  - Works well for simple environments, but struggles with multiple barriers.
  - No booking mechanism, leading to conflicts over high-reward cells.
- Hungarian Booking + A:*

  - Provides a globally optimal assignment.
  - Ensures collision-free movement via a reservation table.
  - Higher computation cost on larger grids or with many agents.
- Custom Lane-Based Booking + A:*

  - Simpler assignment logic using lanes.
  - Near perfect performance in high-density, centralized shapes.
  - Not guaranteed to be globally optimal but performs well in practice.
