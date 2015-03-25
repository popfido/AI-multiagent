# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        foodList = currentGameState.getFood().asList()
        distances = []
        
        # stopping is never a good action
        if action == 'Stop':
            return float('-inf')
        
        for i in range(0, len(newGhostStates)):
            ghostState = newGhostStates[i]
            ghostPos = ghostState.getPosition()
            ghostTime = newScaredTimes[i]
            
            if ghostPos == newPos and ghostTime == 0:
                return float('-inf')
        
        for food in foodList:
            distances.append(-(util.manhattanDistance(newPos, food))) # smaller distances and bigger (less negative) numbers are better
            
        if len(distances) > 0:
            return max(distances)

        return successorGameState.getScore()

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

    def terminalTest(self, currentGameState, depth):
        return (currentGameState.isWin() or currentGameState.isLose())

    def maximizeAgent(self, currentGameState, agentIndex, depth):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("-inf"),Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            newVal = self.minimizeAgent(successor, 1, depth)
            if type(newVal) == tuple:
                newVal = newVal[0]

            if newVal > val[0]:
                val = (newVal, action)

        return val

    def minimizeAgent(self, currentGameState,agentIndex, depth):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("inf"), Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            if agentIndex == currentGameState.getNumAgents() - 1:
                if depth == self.depth:
                    newVal = self.evaluationFunction(successor)
                else:
                    newVal = self.maximizeAgent(successor, 0, depth+1)
            else:
                newVal = self.minimizeAgent(successor,agentIndex+1, depth)

            if type(newVal) == tuple:
                    newVal = newVal[0]

            if newVal < val[0]:
                val = (newVal, action)

        return val 

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
        """
        "*** YOUR CODE HERE ***"

        return self.maximizeAgent(gameState, 0, 1)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def terminalTest(self, currentGameState, depth):
        return (currentGameState.isWin() or currentGameState.isLose())

    def maxPruneAgent(self, currentGameState, agentIndex, depth, alpha, beta):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("-inf"),Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            newVal = self.minPruneAgent(successor, 1, depth, alpha ,beta)
            if type(newVal) == tuple:
                newVal = newVal[0]

            if newVal > beta:
                return (newVal, action)

            if newVal > val[0]:
                val = (newVal, action)

            alpha = max(alpha, val[0])

        return val

    def minPruneAgent(self,currentGameState, agentIndex, depth, alpha, beta):
        if self.terminalTest(currentGameState, depth):
            return self.evaluationFunction(currentGameState)

        val = (float("inf"), Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)

            if agentIndex == currentGameState.getNumAgents() - 1:
                if depth == self.depth:
                    newVal = self.evaluationFunction(successor)
                else:
                    newVal = self.maxPruneAgent(successor, 0, depth+1, alpha, beta)
            else:
                newVal = self.minPruneAgent(successor,agentIndex+1, depth, alpha, beta)

            if type(newVal) == tuple:
                    newVal = newVal[0]

            if newVal < alpha:
                return (newVal, action)
            if newVal < val[0]:
                val = (newVal, action)

            beta = min(beta, val[0])

        return val 

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxPruneAgent(gameState, 0, 1, float("-inf"), float("inf"))[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def terminalTest(self, currentGameState, depth):
        return (currentGameState.isWin() or currentGameState.isLose())

    def maximizeAgent(self, currentGameState, agentIndex, depth):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("-inf"),Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            newVal = self.expectAgent(successor, 1, depth)
            if type(newVal) == tuple:
                newVal = newVal[0]

            if newVal > val[0]:
                val = (newVal, action)

        return val

    def expectAgent(self, currentGameState, agentIndex, depth):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = 0
        prob = 1.0/len(currentGameState.getLegalActions(agentIndex))

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            if agentIndex == currentGameState.getNumAgents() - 1:
                if depth == self.depth:
                    newVal = self.evaluationFunction(successor)
                else:
                    newVal = self.maximizeAgent(successor, 0, depth+1)
            else:
                newVal = self.expectAgent(successor,agentIndex+1, depth)

            if type(newVal) == tuple:
                    newVal = newVal[0]

            val += newVal * prob

        return val

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maximizeAgent(gameState, 0, 1)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The evaluation function begins with the current game score, which helps aintain rewards for going fast and eating ghosts.

      There is a 10 point reward for eating food to encourage pacman to eat food while minus 10 points for every food that still left in the game.

      Also, 200 minus distance to the closest scared ghost score is set to earn points by eating a ghost if they are scared.
      
    """
    "*** YOUR CODE HERE ***"

    foodMatrix = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()
    foodCount = currentGameState.getNumFood()
    currentScore = currentGameState.getScore()

    nearestGhostDistance = float("inf")

    ghostEval = 0
    for ghost in ghostStates:
        ghostPos = (int(ghost.getPosition()[0]), int(ghost.getPosition()[1]))
        dis = manhattanDistance(position, ghostPos)

        if ghost.scaredTimer == 0:
            if dis <= nearestGhostDistance:
                nearestGhostDistance = dis
        else:
            ghostEval += 200 - dis

    if nearestGhostDistance == float("inf"):
        nearestGhostDistance = 0

    ghostEval += nearestGhostDistance

    return currentScore - 10*foodCount + 1*ghostEval

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"

        
        util.raiseNotDefined()
