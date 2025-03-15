# from greedyWeightMap import Environment, Agent,plotState
from collections import deque
from greedyWeightMap import generate_rectangle, plot_shape,generate_cross, generate_diamond,generate_rectangle,hungarianAssignment, Agent,Environment
from matplotlib import pyplot as plt
import numpy as np

# coordinates=generate_cross(10)
# coordinates2= generate_diamond(10)

# coordinates3= generate_rectangle(10)
# for coordinate in coordinates:
#     print(coordinate)
AgentList = []
i=0
for y in range(2):  # First loop over rows
    for x in range(10):  # Then loop over columns
        AgentList.append(Agent(x, 1-y, i))
        i+=1

GRID = Environment(shape='rectangle')


def plotAssignments(agent_assignments):
    GRID_SIZE=10
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, GRID_SIZE)
    ax.set_ylim(-1, GRID_SIZE)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    
    for agent, destination in agent_assignments.items():
        x_coor=agent.x
        y_coor=agent.y
        x_dest=destination[0]
        y_dest=destination[1]
        circle=plt.Circle((x_coor,y_coor),0.3,color='green',alpha=0.8)
        ax.add_patch(circle)
        ax.text(x_coor, y_coor, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
        ax.text(x_dest, y_dest, str(agent.agent_id), ha='center', va='center', fontsize=8, color='black')
        # ax.arrow(x_coor, y_coor, x_dest-x_coor, y_dest-y_coor, head_width=0.1, head_length=0.1, fc='black', ec='black')
    plt.show()
Agent_Assignemnt=hungarianAssignment(AgentList,GRID.shape_positions)
plotAssignments(Agent_Assignemnt)
print(Agent_Assignemnt)
