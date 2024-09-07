# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # python3 pacman.py -l tinyMaze -p SearchAgent
    # python3 pacman.py -l mediumMaze -p SearchAgent
    # python3 pacman.py -l bigMaze -z .5 -p SearchAgent
    from util import Stack
    fringe = Stack()
    start_state = problem.getStartState()
    explored = set()
    start_node = (start_state, [])
    fringe.push(start_node) # start node in stack
    while not fringe.isEmpty():
        current_state, path_so_far = fringe.pop()

        if problem.isGoalState(current_state): # if current state is final 
            # print(path_so_far)
            return path_so_far

        if current_state not in explored: # if not explored before add to explored
            explored.add(current_state)
            for neighbor, action, cost in problem.getSuccessors(current_state): # find its neighbours
                # print((neighbor,action,cost))
                new_path = path_so_far + [action]
                next_node = (neighbor, new_path)
                fringe.push(next_node)
    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # python3 pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
    # python3 pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
    from util import Queue
    fringe = Queue()
    start_state = problem.getStartState()
    explored =set()
    fringe.push((start_state, []))
    while not fringe.isEmpty():
        current_state, path_so_far = fringe.pop()

        if problem.isGoalState(current_state):
            return path_so_far

        if current_state not in explored:
            explored.add(current_state)
            for neighbor, action, cost in problem.getSuccessors(current_state):
                new_path = path_so_far + [action]
                next_node = (neighbor, new_path)
                fringe.push(next_node)

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # python3 pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
    # python3 pacman.py -l mediumDottedMaze -p StayEastSearchAgent
    # python3 pacman.py -l mediumScaryMaze -p StayWestSearchAgent
    from util import PriorityQueue
    fringe = PriorityQueue()
    start = problem.getStartState()
    explored = set()
    start_node = (start, [], 0)
    fringe.push(start_node, 0)
    while not fringe.isEmpty():
        current_state, path_so_far, current_cost = fringe.pop()

        if problem.isGoalState(current_state):
            return path_so_far

        if (current_state not in explored) :
            explored.add(current_state)
            for neighbour, action, cost in problem.getSuccessors(current_state):
                new_cost = current_cost + cost
                new_action = path_so_far + [action]
                next_node = (neighbour, new_action, new_cost)
                fringe.push(next_node, new_cost)

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # python3 pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    from util import PriorityQueue
    fringe = PriorityQueue()
    start_state = problem.getStartState()
    explored = set() # explored set
    state_cost_map = {} # maintain a dict for cost of nodes 
    start_node = (start_state, [], 0) # state, path to that state from start , actual cost
    fringe.push(start_node,0)  # node, priority // Fringe set 
    while not fringe.isEmpty():
        current_state, path_so_far, path_cost = fringe.pop()

        if problem.isGoalState(current_state): # current state is goal
            return path_so_far

        if current_state not in explored: # if not explored 
            explored.add(current_state)  # then add to explored set
            state_cost_map[current_state] = path_cost # update the cost in the dict 
            for neighbor, action, cost in problem.getSuccessors(current_state):
                updated_cost = path_cost + cost
                updated_path = path_so_far+ [action]
                priority = updated_cost + heuristic(neighbor, problem)
                fringe.push((neighbor, updated_path, updated_cost), priority)

    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
