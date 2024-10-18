# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        if successorGameState.isLose():
            score = -99
            return successorGameState.getScore() + score

        if successorGameState.isWin():
            score = 99
            return successorGameState.getScore() + score

        ghostDistance = [
            manhattanDistance(ghost.getPosition(), newPos)
            for ghost in successorGameState.getGhostStates()
            if (ghost.scaredTimer == 0)
        ]

        if len(ghostDistance):
            nearestGhost = min(ghostDistance)
            score -= 1 / nearestGhost

        foodDistance = [manhattanDistance(food, newPos) for food in newFood.asList()]
        nearestFood = min(foodDistance)
        score += 1 / nearestFood

        return successorGameState.getScore() + score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: gameState, depth):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(v, minValue(state.generateSuccessor(0, action), 1, depth))

            return v

        def minValue(state: gameState, agent, depth):

            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = float("inf")

            if agent == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(
                        v, maxValue(state.generateSuccessor(agent, action), depth + 1)
                    )

                return v
            
            for action in legalActions:
                v = min(
                    v,
                    minValue(state.generateSuccessor(agent, action), agent + 1, depth),
                )

            return v

        legalActions = gameState.getLegalActions(0)

        actions = {}
        for action in legalActions:
            actions[action] = minValue(gameState.generateSuccessor(0, action), 1, 0)

        return max(actions, key=actions.get)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: gameState, depth, alpha, beta):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(
                    v,
                    minValue(state.generateSuccessor(0, action), 1, depth, alpha, beta),
                )
                if v > beta:
                    break
                alpha = max(alpha, v)

            return v

        def minValue(state: gameState, agent, depth, alpha, beta):

            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = float("inf")

            if agent == gameState.getNumAgents() - 1:
                for action in legalActions:
                    v = min(
                        v,
                        maxValue(
                            state.generateSuccessor(agent, action),
                            depth + 1,
                            alpha,
                            beta,
                        ),
                    )
                    if v < alpha:
                        break
                    beta = min(beta, v)

                return v

            for action in legalActions:
                v = min(
                    v,
                    minValue(
                        state.generateSuccessor(agent, action),
                        agent + 1,
                        depth,
                        alpha,
                        beta,
                    ),
                )
                if v < alpha:
                    break
                beta = min(beta, v)

            return v

        legalActions = gameState.getLegalActions(0)

        alpha = -float("inf")
        beta = float("inf")
        actions = {}
        for action in legalActions:
            actions[action] = minValue(
                gameState.generateSuccessor(0, action), 1, 0, alpha, beta
            )

            if actions[action] > beta:
                return action
            alpha = max(actions[action], alpha)

        return max(actions, key=actions.get)
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def terminal(state: gameState, depth):

            return state.isWin() or state.isLose() or depth == self.depth

        def maxValue(state: gameState, depth):

            legalActions = state.getLegalActions(0)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = -float("inf")

            for action in legalActions:
                v = max(v, expValue(state.generateSuccessor(0, action), 1, depth))

            return v

        def expValue(state: gameState, agent, depth):
            
            legalActions = state.getLegalActions(agent)

            if terminal(state, depth) or not legalActions:
                return self.evaluationFunction(state)

            v = 0

            for action in legalActions:
                if agent == gameState.getNumAgents() - 1:
                    v2 = maxValue(state.generateSuccessor(agent, action), depth + 1)

                else:
                    v2 = expValue(
                        state.generateSuccessor(agent, action), agent + 1, depth
                    )

                v += v2 / len(legalActions)

            return v

        legalActions = gameState.getLegalActions(0)

        actions = {}
        for action in legalActions:
            actions[action] = expValue(gameState.generateSuccessor(0, action), 1, 0)

        return max(actions, key=actions.get)
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
     #pacman
    pos = currentGameState.getPacmanPosition()
    #food
    foodList = currentGameState.getFood().asList()
    #ghost
    ghostPos = currentGameState.getGhostPosition(1)
    ghostTimer = currentGameState.getGhostStates()[0].scaredTimer
    ghostDis = manhattanDistance(ghostPos, pos)
    #Capsules
    capsules = currentGameState.getCapsules()
    
    #food
    #distance to eat all food.
    foodDis = 99
    for food in foodList:
        foodDis = min(manhattanDistance(pos, food), foodDis)
    foodScore = 530 - len(foodList) * 10 -  foodDis
    
    #ghost
    if ghostTimer > 0:
        ghostScore = max(70 - ghostDis, 62) 
    else:
        ghostScore =-max(70 - ghostDis, 63)

    #Capsules
    capDis = 99
    for c in capsules:
        capDis = min(manhattanDistance(pos, c), capDis)
    capScore =  160 - len(capsules) * 80 - capDis
    score = currentGameState.getScore() + foodScore + ghostScore + capScore

    return  score

# Abbreviation
better = betterEvaluationFunction
