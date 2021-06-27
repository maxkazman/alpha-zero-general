import numpy as np
from scipy.signal import convolve2d

from Game import Game


class MaxTicTacToeGame(Game):
    def __init__(self, n=3, connect=3):
        self.n = n
        self.connect = connect

    def getInitBoard(self):
        b = Board(self.n, self.n)
        return b.state
    
    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n*self.n

    def getNextState(self, board, player, action):
        b = Board(self.n, self.n)
        b.state = np.copy(board)
        b.player_turn = player == 1
        b.make_move((int(action/self.n), action%self.n))
        return (b.state, -player)
    
    def getValidMoves(self, board, player):
        b = Board((self.n, self.n))
        b.state = np.copy(board)
        valids = [0]*self.getActionSize()
        moves = b.get_available_moves()
        if len(moves) == 0:
            return np.asarray(valids)
        for move in moves:
            valids[move[0]*self.n + move[1]] = 1
        return np.asarray(valids)

    def getGameEnded(self, board, player): #assuming player is always 1 here?
        b = Board(self.n, self.n)
        b.state = np.copy(board)
        is_over, who_won = b.game_over()
        if not is_over:
            return 0
        return 1 if who_won else -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board
    
    def getSymmetries(self, board, pi): #not changing
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l
    


class Board():

    def __init__(self, rows=3, cols=3, connect=3, save_moves=False):
        self.ROWS = rows
        self.COLUMNS = cols
        self.connect = connect
        self.state = np.zeros(shape=(self.ROWS,self.COLUMNS))
        self.player_turn = True #true means player 1's turn
        self.save_moves = save_moves
        self.move_stack = []

        #setup kernals
        self.horizontal_kernel = np.ones(shape=(1,self.connect))
        self.vertical_kernel = np.transpose(self.horizontal_kernel)
        self.diag1_kernel = np.eye(self.connect, dtype=np.uint8)
        self.diag2_kernel = np.fliplr(self.diag1_kernel)
        self.detection_kernels = [self.horizontal_kernel, self.vertical_kernel, self.diag1_kernel, self.diag2_kernel]
    
    def get_available_moves(self):
        s = np.where(self.state==0)
        return [(s[0][i], s[1][i]) for i in range(len(s[0]))]
    
    def make_move(self, move): #expects move to be (row, col)
        if self.state[move] != 0:
            raise IndexError("Move is invalid, you dumb butt!!!")
        
        self.state[move] = 1 if self.player_turn else -1
        if self.save_moves:
            self.move_stack.append(move)
        self.player_turn = not self.player_turn

    def check_move(self, move): #doesn't actually change state but returns a copy you can look at
        if self.state[move] != 0:
            raise IndexError("Move is invalid, you dumb butt!!!")
        s = np.array(self.state, copy=True)
        s[move] = 1 if self.player_turn else -1
        return s

    def winning_move(self):

        #if player 1 wins, TRUE
        for kernel in self.detection_kernels:
            if (convolve2d(self.state == 1, kernel, mode="valid") == self.connect).any():
                return True
        
        #if player 2 wins, FALSE
        for kernel in self.detection_kernels:
            if (convolve2d(self.state == -1, kernel, mode="valid") == self.connect).any():
                return False
        
        #if there isn't a winner, None
        return None

    def is_full(self):
        return not (self.state == 0).any()

    def game_over(self):
        #returns a tuple of (if the game is over, who won the game)
        connected = self.winning_move()
        full = self.is_full()

        return ((connected is not None) or full, connected)

    def print(self):
        lines = []
        for i in range(self.ROWS):
            l = '| '
            for j in range(self.COLUMNS):
                piece = 'X' if self.state[i][j] == 1 else 'O' if self.state[i][j] == -1 else '-'
                l += f'{piece} | '
            print(l)

    def reset(self):
        self.state = np.zeros(shape=(self.ROWS,self.COLUMNS))
        self.player_turn = True
        self.move_stack = []