import argparse

from Models import get_baseline_tuner

from Steps import SearchAndTrain, get_threshold, test_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=50,
                        help='data sequence length including output')
    parser.add_argument('--sequence_stride', type=int, default=3,
                        help='stride for starting idx')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size for data loader')

    parser.add_argument('--model_name', type=str, required=True,
                        help='must select what model to execute.')

    args = parser.parse_args()

    if args.model_name == "baseline":
        model, selected_cols, scaler = SearchAndTrain(args, get_baseline_tuner)
        threshold = 0.09991000000000001 # get_threshold(model, selected_cols, scaler, args)
        test_and_save(model, selected_cols, scaler, threshold, args)
    elif args.model_name == "transformer":
        pass


if __name__ == "__main__":
    main()
