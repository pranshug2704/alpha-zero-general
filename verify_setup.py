"""
Simple verification script for Othello Bot with PyTorch.
Tests loading the pretrained model and playing a quick game.
"""
import numpy as np
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import RandomPlayer
from othello.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
import Arena
from utils import dotdict

def test_othello_pytorch():
    print("=" * 50)
    print("Testing Othello with PyTorch...")
    print("=" * 50)

    # Create 6x6 game (matches pretrained model)
    print("\n1. Creating 6x6 Othello game...")
    game = OthelloGame(6)
    print(f"   Board size: {game.n}x{game.n}")
    print(f"   Action space: {game.getActionSize()} possible actions")

    # Create neural network and load pretrained weights
    print("\n2. Loading pretrained PyTorch model...")
    nnet = NNet(game)
    nnet.load_checkpoint('./pretrained_models/othello/pytorch/', '6x100x25_best.pth.tar')
    print("   Model loaded successfully!")

    # Create MCTS with the neural network
    print("\n3. Setting up Monte Carlo Tree Search...")
    args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, args)

    # Create AI player using MCTS
    def ai_player(board):
        return np.argmax(mcts.getActionProb(board, temp=0))

    # Create a random opponent
    random_player = RandomPlayer(game).play

    # Play 2 quick games
    print("\n4. Playing 2 games: AI vs Random Player...")
    arena = Arena.Arena(ai_player, random_player, game)
    results = arena.playGames(2, verbose=False)

    print(f"\n   Results: AI won {results[0]}, Random won {results[1]}, Draws {results[2]}")

    if results[0] > results[1]:
        print("\n✅ SUCCESS: AI player is working and winning games!")
    else:
        print("\n⚠️  AI didn't win all games, but setup is working.")

    print("\n" + "=" * 50)
    print("Verification complete!")
    print("=" * 50)

    return True

if __name__ == "__main__":
    test_othello_pytorch()
