'''
the method and key part of the game
the game requires a 8x8 board
with black and white chess on it
with an ai opponent to play with player
'''

import numpy as np
import enum
import copy
import sys

DEFAULT_WID = 8
NEIGHBOUR = [(-1,-1),(-1,0),(-1,1),
             (0,-1),(0,1),
             (1,-1),(1,0),(1,1)]
#chess type
WHITE = -1  #back hand
BLACK = 1   #front hand
VOID = 0
#player type
HUMAN_MODE = 1
AI_MODE = 2
#ai depth
DEPTH = 1
    
MATRIX = np.array([ [500,-25,10,5,5,10,-25,500],
                    [-25,-45,1,1,1,1,-45,-25],
                    [10,1,3,2,2,3,1,10],
                    [5,1,2,1,1,2,1,5],
                    [5,1,2,1,1,2,1,5],
                    [10,1,3,2,2,3,1,10],
                    [-25,-45,1,1,1,1,-45,-25],
                    [500,-25,10,5,5,10,-25,500]])

    

class AILevel(enum.Enum):
    easy = 0
    medium = 1
    hard = 2
    
class GameResult(enum.Enum):
    not_end = 0
    end    = 1
    win    = 2
    lose   = 3
    #to black

class GameState(enum.Enum):
    waiting = 1
    running = 2
    over    = 3 

#chess board class
class Board:
    def __init__(self, width = DEFAULT_WID, height = DEFAULT_WID, chess_type = BLACK):
        self.width = width
        self.height = height
        self.board = np.zeros((width, height), dtype = int)
        self.board[3:5,3:5] = np.array([[BLACK, WHITE],[WHITE, BLACK]])
        ''' 
        self.board = np.array([[1,1,1,1,1,1,1,-1],
                              [1,1,1,1,1,1,1,-1],
                              [1,1,-1,1,1,1,1,-1],
                              [1,1,1,-1,1,1,1,1],
                              [1,1,-1,1,-1,1,1,1],
                              [1,1,-1,1,-1,-1,1,1],
                              [1,1,1,1,1,-1,1,1],
                              [1,1,1,1,1,-1,1,1]])
        '''
    
        self.valid_dic = { BLACK : set(), WHITE : set() }
        self.updateValidDic(chess_type)
    

    def copyBoard(self, new_board):
        self.width = new_board.width
        self.height = new_board.height
        self.board = copy.deepcopy(new_board.board)
        self.valid_dic = copy.deepcopy(new_board.valid_dic)

    def getBoard(self):
        return self.board


    def getValidList(self, chess_type):
        return self.valid_dic[chess_type]
    

    def setChess(self, chess_type, pos):
        if (self.checkValid(pos, chess_type)):
            self.board[pos[0], pos[1]] = chess_type
            self.reverseChessBoard(chess_type, pos)
            self.updateValidDic(chess_type)
            self.updateValidDic(-chess_type)
            return True
        else:
            print(f'Board::setChess, chess_type={chess_type}, pos={pos}, invalid pos')
        return False
    

    def updateBoard(self, chess_type):
        #reverse chess
        
        #update valid_dic
        ...


    def getResult(self):
        print(f'Board::getResult, black_valid_dic={self.valid_dic[BLACK]}, white_valid_dic={self.valid_dic[WHITE]}')
        if (np.sum(self.board == VOID) > 0 and
            (self.valid_dic[BLACK] or self.valid_dic[WHITE])):
            print(f'result: not_end')
            return GameResult.not_end
        else:
            print(f'result: end')
            return GameResult.end
        

    def calResult(self, result, winner):
        state = None
        black_count = np.sum(self.board == BLACK)
        white_count = np.sum(self.board == WHITE)
        if (black_count > white_count): 
            state = GameResult.win
            winner = BLACK
        elif (black_count < white_count): 
            state = GameResult.lose
            winner = WHITE
        else: 
            state = GameResult.equal
        
        result['state'] = state
        result[BLACK] = black_count
        result[WHITE] = white_count

        print(f'Board::calResult, result={result}, winner={winner}')


    def updateValidDic(self, chess_type):
        valid_set = set()
        #get void pos next to oppnent
        pos_nxt_op = self.getPosNextOpponent(chess_type)
        #print(f'pos_nxt_op: {pos_nxt_op}')
        for p in pos_nxt_op:
            #row/col/diag check if same chess_type exit
            #print(f'pos is {p}')
            if (self.checkForSame(chess_type, p)):
                valid_set.add(p)

        self.valid_dic[chess_type] = copy.deepcopy(valid_set) 
        #print(f'valid set of {chess_type} is:\n{self.valid_dic[chess_type]}')


    def getPosNextOpponent(self, chess_type):
        pos_nxt_op = set()
        op_list = np.argwhere(self.board == -chess_type)
        for op in op_list:
            pos_ard_void = self.getVoidNeighbor(op)
            pos_nxt_op.update(pos_ard_void)

        #print(f'pos_nxt_op: {pos_nxt_op}')
        return pos_nxt_op
        

    def getVoidNeighbor(self, pos):
        neighbour = set()
        for p in NEIGHBOUR:
            x = pos[0] + p[0] 
            y = pos[1] + p[1] 
            if (x < 0 or y < 0 or x > 7 or y > 7):
                continue
            npos = (x, y)
            if (self.board[npos[0], npos[1]] == VOID):
                neighbour.add(npos)
        #print(neighbour)
        return neighbour

    def getStableCount(self, chess_type):
        count = 0
        for i in range(DEFAULT_WID):
            for j in range(DEFAULT_WID):
                if self.checkStable(chess_type, (i, j)):
                    count += 1

        return count

    def checkStable(self, chess_type, pos):
        cpos = list(pos)
        if self.board[cpos[0], cpos[1]] != chess_type:
            return False
        
        for dir in NEIGHBOUR:
            npos = []
            npos.append(cpos[0] + dir[0])
            npos.append(cpos[1] + dir[1])

            if (npos[0] < 0 or npos[0] >= DEFAULT_WID or
                npos[1] < 0 or npos[1] >= DEFAULT_WID):
                continue
            if (self.board[npos[0], npos[1]] == chess_type):
                continue
            else:
                return False
        return True

    def checkForSame(self, chess_type, pos):
        for dir in NEIGHBOUR:
            cpos = list(pos)
            cpos[0] += dir[0]
            cpos[1] += dir[1]
            if (not (cpos[0] >= 0 and cpos[0] < DEFAULT_WID and 
                   cpos[1] >= 0 and cpos[1] < DEFAULT_WID )):
                continue
            if (self.board[cpos[0], cpos[1]] != -chess_type):
                continue

            while (cpos[0] >= 0 and cpos[0] <= 7 and 
                   cpos[1] >= 0 and cpos[1] <= 7):
                chess = self.board[cpos[0], cpos[1]]
                if(chess == VOID):
                    break
                elif(chess == chess_type):
                    return True
                elif(chess == -chess_type):
                    cpos[0] += dir[0]
                    cpos[1] += dir[1]

        return False


    def rvsWithDir(self, board, chess_type, x, y, dir):
        if (x < 0 or x >= DEFAULT_WID or y < 0 or y >= DEFAULT_WID
            or board[x, y] == VOID):
            return False
        
        chess = board[x, y]
        if (chess == chess_type):
            return True

        if (chess == -chess_type):
            rvs = self.rvsWithDir(board, chess_type, x+dir[0], y+dir[1], dir)
            if (rvs == True):
                board[x][y] = chess_type
                return True
            
        return False
    

    def reverseChessBoard(self, chess_type, pos):
        new_board = self.getBoard()
        for dir in NEIGHBOUR:
            cpos = list(pos)
            cpos[0] += dir[0]
            cpos[1] += dir[1]

            self.rvsWithDir(new_board, chess_type, cpos[0], cpos[1], dir)

        self.board = copy.deepcopy(new_board)


    def printBoard(self, chess_type = VOID):
        print(f'valid pos for {chess_type}')

        for i in range(10):
            if (i == 0 or i == 9):
                print('   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |   ')
                print('---------------------------------------')
            else:
                for j in range(10):
                    if (j == 0 ):
                        print(f' {i-1} ', end = '')
                    elif (j == 9):
                        print(f'| {i-1} ', end = '')
                    else:
                        chess = self.board[i-1, j-1]
                        if (chess == BLACK):
                            print('| x ', end = '')
                        elif (chess == WHITE):
                            print('| o ', end = '')
                        elif (chess == VOID):
                            if (chess_type == BLACK or chess_type == WHITE):
                                valid_list = self.getValidPos(chess_type)
                                cur = (i-1, j-1)
                                if cur in self.getValidPos(chess_type):
                                        print('| * ', end = '')
                                else:
                                    print('|   ', end = '')
                            else:
                                print('|   ', end = '')
                       
                print()
                print('---------------------------------------')
        #to be improved


    def getValidPos(self, chess_type):
        return self.valid_dic[chess_type]


    def checkValid(self, pos, chess_type):
        valid_list = self.getValidPos(chess_type)
        if (tuple(pos) in valid_list):
            return True
        else:
            return False
    
    
    #------------------------------evaluate function------------------------------------
        
    def evaluate(self):
        score = 0 
        #topo
        b_topo = self.getChessTopo(BLACK)
        w_topo = self.getChessTopo(WHITE)
        topo_score = float(b_topo - w_topo)
        
        #count
        b_count = np.sum(self.board == BLACK)
        w_count = np.sum(self.board == WHITE)
        count_score = float(b_count - w_count)

        #action
        b_valid_pos = len(self.valid_dic[BLACK])
        w_valid_pos = len(self.valid_dic[WHITE])
        action_score = float(b_valid_pos - w_valid_pos)

        #stable
        b_stable = self.getStableCount(BLACK)
        w_stable = self.getStableCount(WHITE)
        stable_score = float(b_stable - w_stable)

        score  = topo_score 
        return score
    
    
    def getChessTopo(self, chess_type):
        val = 0
        for i in range(DEFAULT_WID):
            for j in range(DEFAULT_WID):
                if (self.board[i, j] == chess_type):
                    val += MATRIX[i, j]
        return val


'''
chess player class

mode: 
1. human player
2. ai player
'''

class Agent:
    def __init__(self, chess_type, mode, para = -1):
        self.chess_type = chess_type
        self.mode       = mode
        self.para       = para     #if mode == ai, para == ai_level
        self.board      = None
        self.depth      = DEPTH

    def setBoard(self, board):
        self.board = board

    def setDepth(self, depth):
        self.depth = depth

    def getChessType(self):
        return self.chess_type
    
    def getNextStep(self, board):
        best_pos = [-1, -1]
        if (self.mode == HUMAN_MODE):
            best_pos = self.getHumanPlayerPos()
        elif (self.mode == AI_MODE):
            dep = self.depth
            self.minMax(board, dep, float('-inf'), float('inf'), self.chess_type, best_pos)

        print(f'Agent::getNextStep, best_pos_address={id(best_pos)}, mode={self.mode}, pos={best_pos}')
        return tuple(best_pos)
        
    def getHumanPlayerPos(self):
        x = int(input('type x:'))
        y = int(input('type y:'))
        if (x < 0 or y < 0 
            or x > DEFAULT_WID - 1 
            or y > DEFAULT_WID - 1):
            return (-1, -1)
        else:
            return (x,y)
        
    def randomSelect():
        ...

    def minMax(self, board, depth, alpha, beta, chess_type, best_pos):

        print('-'*20 + 'in minmax' + '-'*20)
        print(f'Agent::minMax, depth={depth}, alpha={alpha}, beta={beta},' 
              f'chess_type={chess_type}, best_pos={best_pos}, address={id(best_pos)}')

        if depth == 0 or board.getResult() == GameResult.end:
            value = board.evaluate()
            print(f'Agent::minMax, depth={depth}, chess_type={chess_type}, value={value}')
            return value
        
        best = [-1,-1]
        cost = 0
         
        #max
        if (chess_type == self.chess_type):
            print(f'chess_type = {chess_type}!, max')
            cost = float('-inf')
            for pos in board.getValidList(chess_type):
                print(f'pos={pos}')
                nxt_board = Board()
                nxt_board.copyBoard(board)
                nxt_board.setChess(chess_type, pos)
                nxt_board.printBoard(-chess_type)
                val = self.minMax(nxt_board, depth - 1, alpha, beta, -chess_type, best)
                print(f'before, chess_type={chess_type}, pos={pos}, best={best}, val={val}, cost={cost}, alpha={alpha}, beta={beta}')
                if val > cost:
                    best[0] = pos[0]
                    best[1] = pos[1]
                cost = max(cost, val)
                alpha = max(cost, alpha)
                print(f'after, chess_type={chess_type}, pos={pos}, best={best}, val={val}, cost={cost}, alpha={alpha}, beta={beta}')
                if (beta <= alpha): break
        #min
        elif (chess_type != self.chess_type):
            print(f'chess_type = {chess_type}!, min')
            cost = float('inf')
            for pos in board.getValidList(chess_type):
                print(f'pos={pos}')
                nxt_board = Board()
                nxt_board.copyBoard(board)
                nxt_board.setChess(chess_type, pos)
                nxt_board.printBoard(-chess_type)
                val = self.minMax(nxt_board, depth - 1, alpha, beta, -chess_type, best)
                print(f'before, chess_type={chess_type}, pos={pos}, best={best}, val={val}, cost={cost}, alpha={alpha}, beta={beta}')
                if val > cost:
                    best[0] = pos[0]
                    best[1] = pos[1]
                cost = min(cost, val)
                beta = min(cost, beta)
                print(f'after, chess_type={chess_type}, pos={pos}, best={best}, val={val}, cost={cost}, alpha={alpha}, beta={beta}')
                if (beta <= alpha): break
        
        best_pos[0] = best[0]
        best_pos[1] = best[1]
        print(f'Agent::minMax return, chess_type={chess_type}, best_pos={best_pos}, address={id(best_pos)},cost={cost}')
        return cost

    def play(self, board):
        pos = None

        if (self.level == AILevel.easy):
            ...
        elif (self.level == AILevel.medium):
            ...
        elif (self.level == AILevel.hard):
            ...

'''
game class
control the game principle and judge game result
'''

class Game:
    def __init__(self, player = BLACK,  
                 game_state = GameState.waiting):
        self.board  = None
        self.player = player #current player : black / white
        self.winner = None
        self.state  = game_state
        self.bots  = {BLACK : None, WHITE : None}
        self.result = {'state' : 0, BLACK : 0, WHITE : 0}

    def setPlayerMode(self, mode):
        self.player = mode


    def setAIPara(self, para1 = 0, para2 = 0):
        self.para1 = para1
        self.para2 = para2


    def run(self, mode1=HUMAN_MODE, mode2=AI_MODE, para1=AILevel.easy, para2=AILevel.easy):
        self.state = GameState.running
        self.bots[BLACK] = Agent(BLACK, mode = mode1, para = para1)
        self.bots[WHITE] = Agent(WHITE, mode = mode2, para = para2)
        self.board = Board()

        

        #during fighting 
        while (self.state == GameState.running):
            step_result = False
            step_pos = None
            print('\n' * 5)
            print(f'it is {self.player} turn !')
            self.board.printBoard(self.player)

            #get next step of current player
            #set new chess
            #pos_next = self.board.getValidPos(self.player)
            step_pos = self.bots[self.player].getNextStep(self.board)
            step_result = self.setChess(self.player, step_pos)

            #if new step is valid, calc current result
            if(step_result):    
                self.board.printBoard()
                game_result = self.getGameResult()
                print(f'game result is {game_result} !')
                if (game_result != GameResult.not_end):
                    self.state = GameState.over
                    self.calcGameResult()
                    self.showGameResult()
                    break
                else:
                    self.player = -self.player
           

    def setChess(self, player, step_pos):
        return self.board.setChess(player, step_pos)
        

    def getGameResult(self):
        return self.board.getResult()

    def calcGameResult(self):
        self.board.calResult(self.result, self.winner)

    def showGameResult(self):
        print('Game Result:')
        if (self.winner == BLACK):
            print('Winner is: Black Chess')
        elif (self.winner == WHITE):
            print('Winner is: White Chess')
        else:
            print('Equal Game')

        print(f'Scores: Black {self.result[BLACK]} vs White {self.result[WHITE]}')
        

 

def main():
    '''
    board = Board()
    print(board.getResult())
    '''
    f = open('a.log', 'a')
    sys.stdout = f
    sys.stderr = f # redirect std err, if necessary



    game = Game() 
    game.run()

if __name__ == '__main__':
    main()