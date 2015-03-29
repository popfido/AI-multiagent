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
from searchAgents import mazeDistance
import searchAgents
from operator import itemgetter 
from random import randint

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
        
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            ghostTime = ghost.scaredTimer
            
            if ghostPos == newPos and ghostTime == 0:
                return float("-inf")
       
        if action == Directions.STOP:
            return float("-inf")

        for food in foodList:
            distances.append(-(util.manhattanDistance(newPos, food))) # return positive value may cause loop?
            
        if len(distances) > 0:
            # print max(distances)
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

    def terminalTest(self, currentGameState, depth):
        return (currentGameState.isWin() or currentGameState.isLose())

    def generalPacmanAgent(self, currentGameState, agentIndex, depth, alpha, beta, method):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("-inf"),Directions.STOP)

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            newVal = self.generalGhostAgent(successor, 1, depth, alpha, beta, method)
            if type(newVal) == tuple:
                newVal = newVal[0]

            if newVal > beta and method == "AlphaBetaPrune":
                return (newVal, action)

            if newVal > val[0]:
                val = (newVal, action)

            if method == "AlphaBetaPrune":
                alpha = max(alpha, val[0])

        return val

    def generalGhostAgent(self, currentGameState,agentIndex, depth, alpha, beta, method):
        if self.terminalTest(currentGameState,depth):
            return self.evaluationFunction(currentGameState)

        val = (float("inf"), Directions.STOP)

        if (method == "Expectimax"):
            val = 0
            prob = 1.0/len(currentGameState.getLegalActions(agentIndex))

        for action in currentGameState.getLegalActions(agentIndex):
            successor = currentGameState.generateSuccessor(agentIndex, action)
            if agentIndex == currentGameState.getNumAgents() - 1:
                if depth == self.depth:
                    newVal = self.evaluationFunction(successor)
                else:
                    newVal = self.generalPacmanAgent(successor, 0, depth+1, alpha, beta, method)
            else:
                newVal = self.generalGhostAgent(successor,agentIndex+1, depth, alpha, beta, method)

            if type(newVal) == tuple:
                    newVal = newVal[0]

            if newVal < alpha and method == "AlphaBetaPrune":
                return (newVal, action)

            if method != "Expectimax":
                if  type(val) == tuple and newVal < val[0]:
                        val = (newVal, action)

            if method == "AlphaBetaPrune":
                beta = min(beta, val[0])

            if method == "Expectimax":
                val += newVal * prob

        return val 

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
        """
        "*** YOUR CODE HERE ***"

        return self.generalPacmanAgent(gameState, 0, 1, 0, 0, "Minimax")[1]
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
        return self.generalPacmanAgent(gameState, 0, 1, float("-inf"), float("inf"), "AlphaBetaPrune")[1]
        util.raiseNotDefined()

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
        return self.generalPacmanAgent(gameState, 0, 1, 0, 0, "Expectimax")[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Since Q5 using expectimax agent upon evaluationFunction(ghost act randomly), 
      It is reasonable to assume that the ghost acts randomly :)

      Also, the action of pacman looked like that the expectimax agent has been limited to a given depth - important info.

      If we do not consider the depth of the searching tree effect - an action queue of (stop, west) would be treat as the same to (west, stop)
      Thus we cannot stop pacman from stay on one state -- waiting ghost to chase it so that it can move :) WTF
      The way to solve this problem of "Un-preference" -- is to intro some add-in credit to let pacman move.

      First, we aboslutely need to consider the ghost coz eating the scared ghost would get 200 points of reward. 
      Always avoiding from meet with ghost leads usually lead to a score < 1000

      Then capsule would be appreciate while a ghost is getting near to pacman and if the ghost is not scared, 
      having capsules uneaten is bad, so info of capsule is introduced. 

      If the ghost is scared, being close to it is good, and there is no extra bonus (beyond the gameState.getScore() bump for eating a
      food) for being close to a food. In fact, discounting the game state food-eating bump/distance travelled decrement is 
      a good idea so that pacman is sufficiently motivated to chase after the ghost :)
      
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    position = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()
    numCapsules = len(capsules)
    walls = currentGameState.getWalls()
    ghostStates = currentGameState.getGhostStates()
    ghostDistance = []
    foodList = currentGameState.getFood().asList()
    foodDist, numFood = nearestItem(position, foodList, walls)

    for ghost in ghostStates:
        ghostDistance.append(((int((ghost.getPosition()[0])),int((ghost.getPosition()[1]))), manhattanDistance(position, ghost.getPosition()), ghost.scaredTimer))

    if len(ghostDistance) > 0:
        nearestGhost = min(ghostDistance, key=itemgetter(1))
        nearestGhostDistance = mazeDistance(nearestGhost[0], position, currentGameState)
    else: 
        nearestGhostDistance = 100000

    capsuleDistWeight = 3
    capsuleCountWeight = 20
    ghostDistWeight = 40
    foodDistWeight = 0.25
    gameScoreWeight = 1.0

    capsuleDistFeature = 0
    capsuleCountFeature = 0
    ghostDistFeature = 0
    foodDistFeature = 1.0 / foodDist
    gameScoreFeature = currentGameState.getScore()

    if nearestGhost[2]:
        ghostDistFeature = 1.0 / nearestGhostDistance
        foodDistWeight = 0
        gameScoreWeight = 0.99
    elif numCapsules:
        capsuleCountFeature = -1
        capsuleDist, numCapsules = nearestItem(position, capsules, walls)
        capsuleDistFeature = 1.0 / capsuleDist

    utility = (gameScoreWeight * gameScoreFeature +
           capsuleDistWeight * capsuleDistFeature +
           capsuleCountWeight * capsuleCountFeature +
           ghostDistWeight * ghostDistFeature +
           foodDistWeight * foodDistFeature)
    return utility

def nearestItem(position, items, walls):
    if not items:
        return 0, None
    closed = set()
    fringe = util.Queue()
    fringe.push((position, 0))
    while not fringe.isEmpty():
        (x, y), cost = fringe.pop()
        if (x, y) in items:
            return cost, (x, y)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for (nx, ny) in [(x+dx, y+dy) for dx, dy in directions if not walls[x+dx][y+dy]]:
            fringe.push(((nx, ny), cost+1))
    return 0, None

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
        from operator import gt, lt

        def alphabeta(state, alpha, beta, depth):
            if depth == self.depth * state.getNumAgents() or state.isLose() or state.isWin():
                return None, betterEvaluationFunction(state)
            agent = depth % state.getNumAgents()
            OP = gt if agent == 0 else lt
            action, value = None, float('-inf') if agent == 0 else float('inf')
            for a in state.getLegalActions(agent):
                _, v = alphabeta(state.generateSuccessor(agent, a), depth+1, alpha, beta)
                action, value = (a, v) if OP(v, value) else (action, value)
                alpha = max(alpha, value) if agent == 0 else alpha  
                beta = min(beta, value) if agent != 0 else beta     
                if alpha > beta:
                    break
            return action, value

        action, _ = alphabeta(gameState, self.index, float('-inf'), float('inf'))
        return action
        
        util.raiseNotDefined()
