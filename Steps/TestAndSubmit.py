import pandas as pd

from DatasetModules import load_dataset, HAIDataLoader

from settings import TEST_FOLDER, SUBMISSION_PATH, MODEL_SAVE_FOLDER

from .Helpers import test_and_get_list, put_labels


def test_and_save(model, selected_cols, scaler, threshold, args):
    np_test_list = load_dataset(TEST_FOLDER,
                                selected_cols=selected_cols,
                                scaler=scaler)
    test_dataset = HAIDataLoader(np_test_list,
                                 length=args.sequence_length,
                                 stride=1,
                                 batch_size=args.batch_size,
                                 train=False)
    test_result = test_and_get_list(model, test_dataset, np_test_list, args)
    final = put_labels(test_result, threshold)

    submission = pd.read_csv(SUBMISSION_PATH)
    assert len(final) == len(submission), f"final label padding seems wrong: {len(final)} != {len(submission)}"
    submission['attack'] = final
    submission.to_csv(f'{MODEL_SAVE_FOLDER}/{args.model_name}/submission_{args.model_name}.csv', index=False)
    print("#"*20, "ALL PROCESSES ENDED. WISH YOU GOOD LUCK WITH THE RESULT!", "#"*20)

