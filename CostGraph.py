# Example usage with 20 agents
from greedyWeightMap import Environment, Agent,plotState, takeMoveMoore
import time

AgentList = []

for y in range(2):  # First loop over rows
    for x in range(10):  # Then loop over columns
        AgentList.append(Agent(x, 1-y, x + y))

GRID = Environment(shape='rectangle')


if(len(GRID.shape_positions)<len(AgentList)):
    AgentList=AgentList[0:len(GRID.shape_positions)]
    
def run_simulation():
    for i in range(20):
        plotState(GRID,AgentList)
        for agent in AgentList:
            # takeMoveVonNeumann(agent, GRID, AgentList)
            takeMoveMoore(agent, GRID, AgentList)
            print(i)
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()