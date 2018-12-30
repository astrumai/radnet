import argparse
import os
import sys

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "U-Net.trainer"

    from trainer import train, evaluate
    from visualize import interpret

    """ 
    Parse the arguments.
    """

    """see to implement sub parsers for interpret and maybe train and evaluate as needed
    and also for hyperparameter.py """

    parser = argparse.ArgumentParser(description="U-Net.")

    parser.add_argument('--root_dir',
                        default="C:\\Users\\Mukesh\\Segmentation\\U-net\\",
                        type=str,
                        help='root directory'
                        )

    parser.add_argument('--mode',
                        default="train",
                        choices=['train', 'evaluate', 'interpret'],
                        type=str,
                        help='Choose between training and evaluating a trained model'
                        )

    parser.add_argument('--weights_dir',
                        default="./weights",
                        type=str,
                        help='Choose directory to save weights model'
                        )

    parser.add_argument('--log_dir',
                        default="./train_logs",
                        type=str,
                        help='Choose directory to save the logs'
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

    parser.add_argument('--depth',
                        default=3,
                        type=int,
                        help='Number of downsampling/upsampling blocks'
                        )

    parser.add_argument('--n_classes',
                        default=1,
                        type=int,
                        help='Number of classes in the dataset'
                        )

    parser.add_argument('--up_mode',
                        default='upsample',
                        choices=['upconv, upsample'],
                        type=str,
                        help='Type of upsampling'
                        )

    parser.add_argument('--augment',
                        default='yes',
                        choices=['yes, no'],
                        type=str,
                        help='Whether to augment the train images or not'
                        )

    parser.add_argument('--augment_type',
                        default='geometric',
                        choices=['geometric, image, both'],
                        type=str,
                        help='Which type of augmentation to choose from: geometric, brightness or both'
                        )

    parser.add_argument('--transform_prob',
                        default=0.5,
                        type=int,
                        help='Probability of images to augment when calling augmentations'
                        )

    parser.add_argument('--test_size',
                        default=0.2,
                        type=int,
                        help='Validation size to split the data, should be in between 0.0 to 1.0'
                        )

    parser.add_argument('--log',
                        default='no',
                        choices=['yes', 'no'],
                        type=str,
                        help='Log the Values'
                        )

    parser.add_argument('--build_graph',
                        default='no',
                        choices=['yes', 'no'],
                        type=str,
                        help='Build the model graph'
                        )

    parser.add_argument('--plot_interpret',
                        default='block_filters',
                        choices=['sensitivity', 'block_filters'],
                        type=str,
                        help='Type of interpret to plot'
                        )

    parser.add_argument('--interpret_path',
                        default='./visualize',
                        type=str,
                        help='Choose directory to save layer visualizations'
                        )

    parser.add_argument('--plot_size',
                        default=128,
                        type=int,
                        help='Image size of sensitivity analysis'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.interpret_path):
        os.makedirs(args.interpret_path)

    if args.mode == 'train':
        train.train(args)
    elif args.mode == 'evaluate':
        evaluate.evaluate(args)
    elif args.mode == 'interpret':
        interpret.interpret(args)
