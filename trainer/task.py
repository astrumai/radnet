import argparse
import os
import sys

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "U-Net.trainer"

    from trainer import train, evaluate

    """ 
    Parse the arguments.
    """
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="U-Net.")

    parser.add_argument('--root_dir',
                        default="C:\\Users\\Mukesh\\Segmentation\\U-net\\",
                        type=str,
                        help='root directory'
                        )

    parser.add_argument('--mode',
                        default="train",
                        choices=['train', 'evaluate'],
                        type=str,
                        help='Choose between training and evaluating a trained model'
                        )

    parser.add_argument('--weights_dir',
                        default="./weights",
                        type=str,
                        help='Choose directory to save weights model'
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
                        default=5,
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
                        choices=['upconv, upsample'],
                        default='upsample',
                        type=str,
                        help='Type of upsampling'
                        )

    parser.add_argument('--augment',
                        choices=['yes, no'],
                        default='no',
                        type=str,
                        help='Whether to augment the train images or not'
                        )

    parser.add_argument('--augment_type',
                        choices=['geometric, image, both'],
                        default='geometric',
                        type=str,
                        help='Which type of augmentation to choose from: geometric, brightness or both'
                        )

    parser.add_argument('--test_size',
                        default=0.2,
                        type=int,
                        help='Validation size to split the data, should be in between 0.0 to 1.0'
                        )

    args = parser.parse_args()

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    if args.mode == 'train':
        train.train(args)
    elif args.mode == 'evaluate':
        evaluate.evaluate(args)



