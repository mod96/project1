from DatasetModules import load_dataset, HAIDataLoader

from settings import VALIDATION_FOLDER

from .Helpers import test_and_get_list, find_th


def get_threshold(model, selected_cols, scaler, args):
    np_validation_list, attack = load_dataset(VALIDATION_FOLDER,
                                              selected_cols=selected_cols,
                                              scaler=scaler,
                                              ewm=False,
                                              validation=True)
    validation_dataset = HAIDataLoader(np_validation_list,
                                       length=args.sequence_length,
                                       stride=1,
                                       batch_size=args.batch_size,
                                       train=False)
    validation_result = test_and_get_list(model, validation_dataset, np_validation_list, args.sequence_length)

    return find_th(args, attack, validation_result)
