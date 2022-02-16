import argparse

from Models import get_baseline_tuner, get_ViT_tuner, get_ViT_tuner2, get_ViT_tuner3

from Steps import search_and_train, get_threshold, test_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=50,
                        help='data sequence length including output')
    parser.add_argument('--sequence_stride', type=int, default=3,
                        help='stride for starting idx')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size for data loader')

    parser.add_argument('--scaler_type', type=str, default='standard',
                        help='type of scaler.')

    parser.add_argument('--anomaly_smoothing', type=str, default="NO",
                        help='type of smoothing to use : rolling, lfilter, savgol, lowess, search_all')
    parser.add_argument('--smoothing_parameter', type=float, default=0,
                        help='parameter of smoothing')

    parser.add_argument('--range_check_window', type=int, default=0,
                        help='if 0, no range_check.')

    parser.add_argument('--model_name', type=str, required=True,
                        help='must select what model to execute.')

    args = parser.parse_args()

    if args.anomaly_smoothing not in ["NO", "search_all"]:
        assert int(args.smoothing_parameter) != 0, "if selected smoothing, must give param."

    if args.model_name == "baseline":
        model, selected_cols, scaler = search_and_train(args, get_baseline_tuner)
        threshold = get_threshold(model, selected_cols, scaler, args)
        test_and_save(model, selected_cols, scaler, threshold, args)
    elif args.model_name == "transformer":
        model, selected_cols, scaler = search_and_train(args, get_ViT_tuner,
                                                        timelen=args.sequence_length)
        threshold = get_threshold(model, selected_cols, scaler, args)
        test_and_save(model, selected_cols, scaler, threshold, args)
    elif args.model_name == "transformer2": # python main.py --scaler_type minmax --model_name transformer2 --sequence_length 100 --batch_size 128
        model, selected_cols, scaler = search_and_train(args, get_ViT_tuner2,
                                                        timelen=args.sequence_length)
        threshold = get_threshold(model, selected_cols, scaler, args)
        test_and_save(model, selected_cols, scaler, threshold, args)
    elif args.model_name == "transformer3": # python main.py --scaler_type minmax --model_name transformer2 --sequence_length 100 --batch_size 128
        model, selected_cols, scaler = search_and_train(args, get_ViT_tuner3,
                                                        timelen=args.sequence_length)
        threshold = get_threshold(model, selected_cols, scaler, args)
        test_and_save(model, selected_cols, scaler, threshold, args)

if __name__ == "__main__":
    main()
