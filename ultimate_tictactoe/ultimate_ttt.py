import numpy as np
import itertools
from scipy.signal import convolve2d

from Game import Game


class UltimateTTT(Game):
    def __init__(self):
        self.n = 3 #no other option for now
    
    def getInitBoard(self):
        b = Board()
        return b

    def getBoardSize(self):
        return (self.n*self.n, self.n*self.n)

    def getActionSize(self):
        return self.n**4 #3^4 = 81

    def getNextState(self, board, player, action):
        # b = Board()
        # b.state = np.copy(board)
        # b.update_meta()
        board.make_move((int(action/9), action%9), player)
        return (board, -player)

    def getValidMoves(self, board, player):
        b = Board()
        b.state = np.copy(board)
        b.update_meta()
        valids = [0]*self.getActionSize()
        moves = b.get_available_moves()
        if len(moves) == 0:
            print('we ever get in here?')
            return np.asarray(valids)
        for move in moves:
            valids[move[0]*9 + move[1]] = 1
        return np.asarray(valids)

    def getGameEnded(self, board, player):
        # b = Board()
        # b.state = np.copy(board)
        # b.update_meta()
        return board.is_game_over()

    def getCanonicalForm(self, board, player):
        return player*board.state

    def getSymmetries(self, board, pi): #not changing
        # mirror, rotational
        assert(len(pi) == self.n**4)  # 1 for pass
        pi_board = np.reshape(pi[:], (self.n**2, self.n**2))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def evaluate_board(self, board):
        # b = Board()
        # b.state = np.copy(board)
        # b.meta_state = np.copy(meta)
        return board.evaluate()

    @staticmethod
    def display(board):
        # b = Board()
        # b.state = board
        board.display()

class Board():
    def __init__(self):
        self.ROWS, self.COLUMNS, self.connect = 3, 3, 3
        self.state = np.zeros(shape=(self.ROWS*3,self.COLUMNS*3))
        self.meta_state = np.zeros(shape=(self.ROWS, self.COLUMNS))

        #setup kernals
        self.horizontal_kernel = np.ones(shape=(1,self.connect))
        self.vertical_kernel = np.transpose(self.horizontal_kernel)
        self.diag1_kernel = np.eye(self.connect, dtype=np.uint8)
        self.diag2_kernel = np.fliplr(self.diag1_kernel)
        self.detection_kernels = [self.horizontal_kernel, self.vertical_kernel, self.diag1_kernel, self.diag2_kernel]
        
    def get_available_moves(self):
        if (self.state==0).all():
            return list(itertools.product(range(self.ROWS*self.COLUMNS), repeat=2))
        x, y = self.convert_coord(tuple(np.argwhere(np.absolute(self.state)==2)[0]))
        minigame = self.state[x*3:(x+1)*3, y*3:(y+1)*3]
        if self.minigame_is_over((x,y)):
            #do every other miniboard
            boards = list(np.argwhere(self.meta_state==0))
            possible_moves = []
            for board in boards:
                x, y = board[0], board[1]
                minigame = self.state[x*3:(x+1)*3, y*3:(y+1)*3]
                moves = list(np.argwhere(minigame==0))
                moves = [(m[0]+(x*3), m[1]+(y*3)) for m in moves]
                possible_moves.extend(moves)
        else:
            possible_moves = list(np.argwhere(minigame==0))
            possible_moves = [(m[0]+(x*3), m[1]+(y*3)) for m in possible_moves]
            
        return possible_moves
        
    def make_move(self, move, player): #player is either 1 or -1
        if move not in self.get_available_moves():
            raise ValueError('INVALID MOVE YOU DUMB BUTT!')
        #change the last move to not the most recent
        self.state[np.absolute(self.state)==2] /= 2
        
        #update new move
        self.state[move[0], move[1]] = 2*player
        
        #update metaboard if needed
        meta_coord = (int(move[0]/3), int(move[1]/3))
        minigame_result = self.minigame_is_over(meta_coord)
        if minigame_result != 0:
            self.meta_state[meta_coord[0], meta_coord[1]] = minigame_result
    
    def update_meta(self):
        for x in range(3):
            for y in range(3): #get all possible meta coords
                minigame_result = self.minigame_is_over((x, y))
                if minigame_result != 0:
                    self.meta_state[x, y] = minigame_result   

    def minigame_is_over(self, coord): #takes minigame coord
        x, y = coord
        minigame = np.copy(self.state[x*3:(x+1)*3, y*3:(y+1)*3])
        minigame[np.absolute(minigame)==2] /= 2
        return self.check_convolve(minigame)
    
    def is_game_over(self):
        return self.check_convolve(self.meta_state)
    
    def check_convolve(self, game):
        for kernel in self.detection_kernels:
            if (convolve2d(game == 1, kernel, mode="valid") == self.connect).any():
                return 1
    
        for kernel in self.detection_kernels:
            if (convolve2d(game == -1, kernel, mode="valid") == self.connect).any():
                return -1
            
        if (game != 0).all():
            return 1e-4 #full and draw
        
        return 0 #can still play
    
    def evaluate(self):

        score = 0
        for i in range(self.connect**2):
            row = i//3
            col = i%3
            if self.meta_state[row, col] == 0:
                mini_game = np.copy(self.state[row*3:(row+1)*3, col*3:(col+1)*3])
                mini_game[np.absolute(mini_game) == 2] /= 2
                score += self.evaluate_subgame(mini_game, 1) - self.evaluate_subgame(mini_game, -1)
                #print(f'score for {row}, {col}: {self.evaluate_subgame(mini_game, 1) - self.evaluate_subgame(mini_game, -1)}')
                

        score += 10*self.evaluate_subgame(self.meta_state, 1) - 10*self.evaluate_subgame(self.meta_state, -1)
        return score
            

    def evaluate_subgame(self, game, player):
        score = 0
        for kernal in self.detection_kernels:
                sub_score_1 = convolve2d(game == player, kernal, mode="valid") #optionally *2 this
                sub_score_0 = convolve2d(game == 0, kernal, mode="valid")
                #print(f'subscore1: {sub_score_1}')
                #print(f'subscore0: {sub_score_0}')
                combined_score = sub_score_1[sub_score_1+sub_score_0 == 3]
                #print(combined_score)
                score += np.sum(combined_score)
        return score

    @staticmethod
    def convert_coord(coord):
        x, y = coord
        return x%3, y%3
    
    def display(self):
        pieces = {
            0: ' ',
            1: 'x',
            -1: 'o',
            2: 'X',
            -2: 'O'
        }
        for i, row in enumerate(self.state):
            l = '|'
            for j, piece in enumerate(row):
                l += f'{pieces[piece]}|'
                l += '|' if j==2 or j==5 else ''
            if i==3 or i==6:
                print('='*21)
            print(l)

# b = Board()
# arr = np.zeros((9,9))
# arr[0,0] = 1
# arr[0,1] = 1
# arr[0,2] = -1
# arr[2, 1] = -1
# arr[1,1] = 1
# arr[2,2] = 1
# arr[6,6] = -1
# print(arr)
# b.state = arr
# print(b.evaluate())
