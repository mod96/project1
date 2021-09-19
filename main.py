import argparse


def main():
    """
    # data options

    sequence_length
    sequence_stride
    batch_size
    generator_option

    # model options # TODO

    model_name = baseline
                 transformer
                 ...            => 결정하면 load, search, train, find th, test and save까지 다.
                                    중요한건 중간에 runtime 끊겨도 다시 시작하면 repeat 없게.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_length', type=int, default=50, help='data sequence length including output')
    parser.add_argument('--sequence_stride', type=int, default=3, help='stride for starting idx')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--generator_option', type=int, default=1, help='1 or 2')

    args = parser.parse_args()

    # TODO: train


    # TODO: validation and find threshold


    # TODO: test and save submission


if __name__ == "__main__":
    main()
