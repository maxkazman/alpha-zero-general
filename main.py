import logging

import coloredlogs

from Coach import Coach
from ultimate_tictactoe.ultimate_ttt import UltimateTTT
from ultimate_tictactoe.NNet import NNetWrapper as nn
# from max_tictactoe.tictactoe import MaxTicTacToeGame
# from max_tictactoe.NNet import NNetWrapper as nn
# from tictactoe.TicTacToeGame import TicTacToeGame
# from tictactoe.keras.NNet import NNetWrapper as nn
# from othello.OthelloGame import OthelloGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 1000,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.53,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 100000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 16,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 100,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.75,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', UltimateTTT.__name__)
    g = UltimateTTT()
    #g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
