#### CS 441, Fall 2022 - Program #3 - 11/30/22 - Dan Jang
#### Implement a Reinforcement Learning based Agent that will solve Problem #1, which is of Robot Navigation, through the programming of "Robby the Robot!"

from matplotlib import pyplot
import enum as e
import numpy as np
import random as rnd
import time

### I. Global Constants & Initialization

GRID = 10
# Assuming a square grid of dimensions, GRID x GRID
CAN_RATE = 0.5
# 0.5 = 50% chance of can-tile (as well as 50% chance of non-can-tile)

TRAINING_PLOT_POINTS = 100
TESTING_PLOT_POINTS = 100

### II. Specifications of the Q-Learning Algorithmic Initialization-Configuration

N_EPISODES = 5000
M_STEPS = 200
ETA_LEARNING_RATE = 0.2
GAMMA_DISCOUNT_FACTOR = 0.9


### III. Epsilon-Type Greedy Action Selection ("EPSGAS") Initialization-Configuration

EPSGAS_GREED_FACTOR = 0.1
EPSGAS_BASELINE = 0.1 # to be equal to EPSGAS_GREED_FACTOR as an default-stored value of the initial epsilon-greedy factor
EPSGAS_STEP = 0.002
EPSGAS_EPISODE_INTERVAL = 50


### IV. Algorithmic, Agent, & Environmental Helper Functions & Classes

class AgentState(e.IntEnum):
    
    CLEAN_TILE = 0
    CAN_TILE = 1
    WALL_TILE = 2


class RoboBank(e.IntEnum):
    
    CAN_REFUND = 10 # Robby receives a reward of 10 Hypothetical RoboCoins (or any arbitrary denomination / concept of value) for each can recycled (picked-up)
    OOPS = -5 # Robby receives a (anti-)reward of -5 RoboCoins for bumping into a wall
    WHERE_CAN = -1 # Robby receives a (anti-)reward of -1 RoboCoins for attempting to pick-up an non-existant can (Robby has to purchase a 1 RoboCoin priced can-scoop cleaning wipe to disinfect their can-scooper)
    NORMAL_MOVEMENT = 0 # No reward or anti-reward is given to Robby for normal movement that is not specified as above, yay!


### V. Robby the Robot Algorithmic Implementation, Functions, & Class(es)
    
class RobbyTheRobot:
    
    def __init__(self):
        self.grid = self.gridualize()
        self.column, self.row = self.randompos()
        self.robocoins = [] # Integer value of reward(s) and/or anti-reward(s) received by Robby the Robot during each Episode
        self.qmatrix = np.zeros((3, 3, 3, 3, 3, 5)) # Five rows with three different AgentStates defined in that class & 5 to represent the number of columns & the number of actions         #np.zeros((GRID, GRID, 5)) # Qmatrix where GRID x GRID x 5
        
        # Bonus Constant for Total RoboCoins Robby the Robot has collected from all episodes & steps - both from testing & testing combined - together! (Purely for fun!)
        self.robobankbal = 0
    
    
    @staticmethod
    def gridualize():
        
        g_size = GRID + 2
        grid = np.zeros((g_size, g_size))
        
        for x in range(g_size):
            for y in range(g_size):
                
                if x == 0 or x == g_size - 1 or \
                    y == 0 or y == g_size - 1:
                    grid[x][y] = AgentState.WALL_TILE
                    
                else:
                    if rnd.uniform(0, 1) <= CAN_RATE:
                        grid[x][y] = AgentState.CAN_TILE
                
        return grid
    
    
    @staticmethod
    def randompos():
        
        #return rnd.randint(1, GRID), rnd.randint(1, GRID)
        return rnd.randrange(GRID) + 1, rnd.randrange(GRID) + 1
    
    
    #def robbyreboot(self):
    
        # self.grid = self.gridualize()
        # self.column, self.row = self.randompos()
        # self.robocoins = []
    
    
    def robodeposit(self):
        self.robobankbal += sum(self.robocoins)
        self.robocoins = [] # Robby deposits their money from training & testing into a vault [resets per intertraining & intertesting phases of the algorithmic learning process-programming]
    
    ## V.1. Robby the Robot's Action Selection Algorithmic Implementation: Movement & Action Function-Implementations
    
    
    def sense(self, type):
        
        # A. If Type = 1: Current Sense
        if type == 1:
            return int(self.grid[self.column][self.row])
        
        # B. If Type = 2: North Sense
        elif type == 2:
            return int(self.grid[self.column][self.row - 1])
        
        # C. If Type = 3: South Sense
        elif type == 3:
            return int(self.grid[self.column][self.row + 1])
        
        # D. If Type = 4: East Sense
        elif type == 4:
            return int(self.grid[self.column + 1][self.row])
        
        # E. If Type = 5: West Sense
        elif type == 5:
            return int(self.grid[self.column - 1][self.row])
        
        
    def act(self, type):

        # A. If type = 0: Robby the Robot attempts to move Northwise!
        if type == 0:
            if self.sense(2) == AgentState.WALL_TILE:
                return RoboBank.OOPS # Robby the Robot oopsies-daisy's into a wall-tile attempting to move Northwise!
            
            else:
                self.row -= 1
                return RoboBank.NORMAL_MOVEMENT # Robby the Robot successfully moves Northwise normally!
            
        # B. If type = 1: Robby the Robot attempts to move Southwise!
        elif type == 1:
            if self.sense(3) == AgentState.WALL_TILE:
                return RoboBank.OOPS # Robby the Robot oopsies-daisy's into a wall-tile attempting to move Southwise!
            
            else:
                self.row += 1
                return RoboBank.NORMAL_MOVEMENT # Robby the Robot successfully moves Southwise normally!
        
        # C. If type = 2: Robby the Robot attempts to move Eastwise!
        elif type == 2:
            if self.sense(4) == AgentState.WALL_TILE:
                return RoboBank.OOPS # Robby the Robot oopsies-daisy's into a wall-tile attempting to move Eastwise!
            
            else:
                self.column += 1
                return RoboBank.NORMAL_MOVEMENT # Robby the Robot successfully moves Eastwise normally!
            
        # D. If type = 3: Robby the Robot attempts to move Westwise!
        elif type == 3:
            if self.sense(5) == AgentState.WALL_TILE:
                return RoboBank.OOPS # Robby the Robot oopsies-daisy's into a wall-tile attempting to move Westwise!
            
            else:
                self.column -= 1
                return RoboBank.NORMAL_MOVEMENT # Robby the Robot successfully moves Westwise normally!
            
        # E. If type != {0:3}: Robby the Robot attempts to pick-up a can!
        #if type == 5:
        else:
            if self.sense(1) == AgentState.CAN_TILE:
                self.grid[self.column][self.row] = AgentState.CLEAN_TILE
                return RoboBank.CAN_REFUND # Robby the Robot successfully picks-up a can!
            
            else:
                return RoboBank.WHERE_CAN # Robby the Robot attempts to pick-up a can that does not exist, oh dear!
            
            
    ## V.2. Robby the Robot's Main AI Algorithmic Programming: Epsilon-Type Greedy Action Selection ("EPSGAS"), Q-Learning Based Reinforcement Learning Algorithmic Function-Implementations        

    def qgen(self, state, qaction):
        
        return self.qmatrix[int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(qaction)]
    
    
    def qset(self, state, action, qvalue):
        
        self.qmatrix[int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(action)] = qvalue


    def epsgas(self, state, episode, testmode):
        
        if testmode == True:
            eps = EPSGAS_BASELINE
        else:
            eps = EPSGAS_GREED_FACTOR
            eps -= int(episode / EPSGAS_EPISODE_INTERVAL) * EPSGAS_STEP
            
        if eps > rnd.uniform(0, 1):
            epsaction = rnd.randrange(0, 5)
        else:
            epsaction = self.strategicaction(state)
            
        return epsaction
    
    
    def strategicaction(self, state):
        
        action_prices = np.zeros(5)
        
        for idx in range(len(action_prices)):
            action_prices[idx] = self.qgen(state, idx)
            
        return np.argmax(action_prices)
    
    
    def scan(self):
        
        state = np.zeros(5) # 5 different states that Robby the Robot can be in at any given time
        
        state[0] = self.sense(1) # Robby the Robot's current state
        state[1] = self.sense(2) # Robby the Robot's North state
        state[2] = self.sense(3) # Robby the Robot's South state
        state[3] = self.sense(4) # Robby the Robot's East state
        state[4] = self.sense(5) # Robby the Robot's West state
        return state
    
    
    def tstep(self, episodeid, testmode=False):
        
        stepstate = self.scan()
        stepaction = self.epsgas(stepstate, episodeid, testmode)
        steprobowallet = self.act(stepaction)
        
        nextstate = self.scan()
        
        if testmode == False: # If Robby the Robot is NOT in testing mode, therefore, if Robby the Robot is instead, in training mode, then the Q-Learning Algorithmic Function-Implementations are used to update the Q-Matrix so that Robby the Robot can learn from their experiences, awesome!
            qmatrix = self.qgen(stepstate, stepaction)
            maximum_a_qmatrix = self.qgen(nextstate, self.strategicaction(nextstate))
            
            new_qmatrix = qmatrix + ETA_LEARNING_RATE * (steprobowallet + (GAMMA_DISCOUNT_FACTOR * maximum_a_qmatrix) - qmatrix)
            self.qset(stepstate, stepaction, new_qmatrix)
            
        return steprobowallet
            
    
    def episoda(self, episodeid, testmode=False):
        
        robowallet = 0
        
        for idx in range(M_STEPS):
            robowallet += self.tstep(episodeid, testmode)
        
        self.robocoins.append(robowallet)
        self.grid = self.gridualize() # Re-generating the Grid each episode!
        self.column, self.row = self.randompos() # Randomized positions each time!
        return robowallet
    
    
### VI. The Main Program Implementation of Robby the Robot!

def main():
    print("./robolearn.AI: Welcome to RoboLearnOS v3.0!\n")
    time.sleep(3)
    print("./robolearn.AI: Robby the Robot is now booting up...")
    time.sleep(5)
    
    robby = RobbyTheRobot()
    print("./robolearn.AI: Robby the Robot has successfully booted up!")
    time.sleep(2)
    print("\n./robolearn.AI: Here is the [Initial Grid] composed of three, different types of square tiles, that Robby the Robot has been randomly placed within...")
    time.sleep(5)
    print("./robolearn.AI: ...where '0'-tiles are [Clean Tiles], '1'-tiles are [Can Tiles], and '2'-tiles are [Wall Tiles] as follows:")
    time.sleep(7)
    print("")
    print(robby.grid)
    print("")
    time.sleep(10)
    print("./robolearn.AI: ...where Robby the Robot's randomly-placed tile is currently located at the following (x,y) coordinates:")
    time.sleep(1)
    print("\n(" + str(robby.column) + ", " + str(robby.row) + ")\n")
    time.sleep(5)
    # A. Robby the Robot's Environment is initialized!
    print("./robolearn.AI: ...Alright, sweet! Robby the Robot is now [ready to learn!]")
    time.sleep(2)
    # B.I. Robby the Robot Begins Training!
    print("./robolearn.AI: Robby the Robot will soon [begin training!]")
    time.sleep(2)
    print("./robolearn.AI: Creating a blank training-graph...")
    x1points = []
    y1points = []
    print("./robolearn.AI: ...new training-graph successfully created!")
    
    # B.II. Robby the Robot Performs RL-Based Trainings for a specified number of episodes!
    print("\n\n./robolearn.AI: Robby the Robot will commence training in 3...")
    time.sleep(1)
    print("./robolearn.AI: 2..")
    time.sleep(1)
    print("./robolearn.AI: 1.")
    time.sleep(1)
    print("\n./robolearn.AI: [Begin Training!]\n")
    time.sleep(1)
    for trainepisodes in range(N_EPISODES):
        robobal = robby.episoda(trainepisodes)
        print("./robolearn.AI: Robby the Robot has performed [Training] Episode #" + str(trainepisodes) + ", with " + str(robobal) + " RoboCoins!")

    # C. Robby the Robot Completes Testing!
    print("")
    print("./robolearn.AI: Robby the Robot has [successfully completed training!]")
    time.sleep(1)
    
    # D. Formulating the Training-Reward-Graph!
    print("\n")
    print("./robolearn.AI: The RoboAccount Firm ACME Inc. Corp. is now tabulating Robby the Robot's [training] results!")
    time.sleep(1)
    print("\n\n")
    print("<!> ./robolearn.AI: Please make sure to close the displayed [Training-Reward-Graph] to continue! <!>")

    for x1 in range(int(len(robby.robocoins) / TRAINING_PLOT_POINTS)):
        x1point = x1 * TRAINING_PLOT_POINTS
        y1point = 0
        
        for y1 in range(TRAINING_PLOT_POINTS):
            y1point += robby.robocoins[x1 * TRAINING_PLOT_POINTS + y1]
        
        y1point /= TRAINING_PLOT_POINTS
        
        x1points.append(x1point)
        y1points.append(y1point)
    
    pyplot.plot(x1points, y1points, label="Robby the Robot's [Training-Reward] Plot!")
    pyplot.title('Robby the Robot: [Training-Reward] Plot!')
    pyplot.xlabel('# of Training Episode Iterations')
    pyplot.ylabel('RoboCoins (reward-values)')
    pyplot.legend()
    pyplot.xlim([TRAINING_PLOT_POINTS, N_EPISODES])
    pyplot.ylim([-100, 700])
    
    pyplot.show()
    # pyplot.savefig('robby-training-reward-plot.png')
    
    pyplot.close()
    
    print("\n./robolearn.AI: Awesome, thank you!")
    time.sleep(2)
  
    # E.I. Robby the Robot Begins Testing!
    print("\n./robolearn.AI: Robby the Robot will soon [begin testing!]")
    
    print("./robolearn.AI: Creating a blank testing-graph...")
    x2points = []
    y2points = []
    print("./robolearn.AI: ...new testing-graph initialized successfully!")
    
    # E.II. Resetting Robby the Robot's RoboCoin Transaction History for Testing!
    print("./robolearn.AI: Depositing Robby the Robot's Training-Compensation RoboCoins into their RoboBank Account!\n\t(Dan's Dev Comment: Actual used RoboCoins variable will still be reset correctly.)")
    robby.robodeposit()
    time.sleep(1)
    print("./robolearn.AI: Robby the Robot's RoboBank Account has been successfully credited with their Training-Compensation RoboCoins!")
    print("./robolearn.AI: Robby the Robot's RoboWallet Transaction History is now: ", robby.robocoins, " RoboCoins!")
    time.sleep(1)
    # E.III. Robby the Robot Performs RL-Based Testings for a specified number of episodes!
    print("\n\n./robolearn.AI: Robby the Robot [Testing] will commence in 3...")
    time.sleep(1)
    print("./robolearn.AI: 2..")
    time.sleep(1)
    print("./robolearn.AI: 1.")
    time.sleep(1)
    print("\n./robolearn.AI: [Begin Testing!]\n")
    time.sleep(1)
    for testepisodes in range(N_EPISODES):
        robobal = robby.episoda(trainepisodes, True)
        print("./robolearn.AI: Robby the Robot has performed [Testing] Episode #" + str(testepisodes) + ", with " + str(robobal) + " RoboCoins!")

    # F. Robby the Robot Completes Testing!
    print("")
    print("./robolearn.AI: Robby the Robot has [successfully completed testing!]")
    time.sleep(3)
    
    # G. Formulating the Testing-Graph!
    print("\n")
    print("./robolearn.AI: The RoboAccount Firm ACME Inc. Corp. is now tabulating Robby the Robot's [testing] results!")
    time.sleep(1)
    print("\n\n")
    print("<!> ./robolearn.AI: Please make sure to close the displayed [Testing-Graph] to continue! <!>")
    
    for x2 in range(int(len(robby.robocoins) / TESTING_PLOT_POINTS)):
        x2point = x2 * TESTING_PLOT_POINTS
        y2point = 0
        
        for y2 in range(TESTING_PLOT_POINTS):
            y2point += robby.robocoins[x2 * TESTING_PLOT_POINTS + y2]
        
        y2point /= TESTING_PLOT_POINTS
        
        x2points.append(x2point)
        y2points.append(y2point)
    
    pyplot.plot(x2points, y2points, label="Robby the Robot's [Testing-Data] Plot!")
    pyplot.title('Robby the Robot: [Testing-Data] Plot!')
    pyplot.xlabel('# of Testing Episode Iterations')
    pyplot.ylabel('RoboCoins (reward-values)')
    pyplot.legend()
    
    pyplot.xlim([TESTING_PLOT_POINTS, N_EPISODES])
    pyplot.ylim([-100, 700])
    
    pyplot.show()
    # pyplot.savefig('robby-testing-reward-plot.png')
    
    pyplot.close()
    
    print("\n./robolearn.AI: Thank you, awesome!")
    time.sleep(2)

    # H. Robby the Robot's RoboCoin Transaction History is now displayed! ("Just for Fun" Statistic Based on: Training + Testing)
    print("\n\n")
    print("./robolearn.AI: Here is Robby the Robot's [Grand RoboBank Balance]...")
    print("./robolearn.AI: ...which represents their [Total Accumulated Sum of RoboCoins] throughout [both Training & Testing]:")
    
    print("\n..." + str(robby.robobankbal) + " RoboCoins!\n")
    time.sleep(1)
    
    # I. Conclusion with Robby the Robot's Test-Average & Test-Standard-Deviation values!
    print("\n\n")
    print("./robolearn.AI: Robby the Robot's [[Conclusion Statistics]] are as follows...")
    time.sleep(5)
    testaverage = sum(y2points) / len(y2points)
    print("\n./robolearn.AI: ...Robby the Robot's [Test-Average] value is: ", testaverage)
    time.sleep(5)
    teststandarddeviation = np.std(y2points)
    print("\n./robolearn.AI: ...Robby the Robot's [Test-Standard-Deviation] value is: ", teststandarddeviation)
    time.sleep(10)
    # J. That's all folks! Thank you for using Robby the Robot!
    print("\n\n./robolearn.AI: Robby the Robot has [successfully completed] all of its tasks!")
    time.sleep(10)
    print("\n./robolearn.AI: ...well, there is always room for improvement!\n")
    time.sleep(2)
    print("./robolearn.AI: ...but for now, that's all folks!")
    time.sleep(4)
    print("\n")
    print(r"""\ cs
 \ 441  oo
  \____|\mm
  //_//\ \_\
 / rl/AI\/_/
/___/_____\
-----------
""")
    print("")
    print("./robolearn.AI: Thank you for using RoboLearnOS!\n")
    time.sleep(3)
    print("./robolearn.AI: Please have a lovely Winter Break!\n")
    time.sleep(5)
    print("./robolearn.AI: [Program is now wrapping-up...]\n")
    time.sleep(3)
    
if __name__ == "__main__":
    
## Credits to ASCIIWorld.com (http://www.asciiworld.com/-Robots,24-.html) for the ASCII art below:
    print("")
    print(r"""\ cs
 \ 441  oo
  \____|\mm
  //_//\ \_\
 / rl/AI\/_/
/___/_____\
-----------
""")
    time.sleep(1)
    print("RoboLearnOS v3.0.0, by dan & co. (llc) nov. 2022 - Special Thanks to TA Li-Yun & Dr. Rhodes!")
    print("")
    time.sleep(2)
    
    
    main()
    
    
