from src.train.trainer import Trainer
from src.game.game import Game
from src.model.model import OthelloModel

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

args_dict = {
    'size': 6, # size of the board
    # TODO: add arguments
}

args = Args(**args_dict)

def main():
    game = Game(args.size)

    model = OthelloModel(game)

    if args.load_model:
        # TODO: change file and folder names
        model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    trainer = Trainer(game, model, args)

    if args.load_model:
        trainer.loadTrainExamples()

    trainer.learn()

if __name__ == "__main__":
    main()