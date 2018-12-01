# Sample code below:

import argparse
import os
import sys

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    # __package__ = "UnetWork.trainer"

    from ..trainer import train

    """ 
    Parse the arguments.
    """
    # python task.py dynamic_routing

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="U-Net.")

    parser.add_argument('root_dir',
                        type=str,
                        help='root directory'
                        )

    parser.add_argument('--image_size',
                        default=64,
                        type=int,
                        help='resize image size'
                        )

    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='batch size'
                        )

    parser.add_argument('--epochs',
                        default=2,
                        type=int
                        )

    parser.add_argument('--transform',
                        choices=['yes, no'],
                        default='yes',
                        type=str,
                        help=' Whether to add transformations to the images'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train.train(args)



