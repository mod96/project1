import glob

from DatasetModules import load_dataset, train_valid_split, HAIDataLoader

from settings import TRAIN_FOLDER, MAX_EPOCH, MODEL_SAVE_FOLDER, EARLY_STOP_PATIENCE

from Callbacks import ClearTrainingOutput, CheckpointSave, EarlyStopAndSave


def SearchAndTrain(args, get_tuner):
    train_dataset, valid_dataset, selected_cols, scaler = prepare_train_datasets(args)
    n_features = len(selected_cols)

    tuner = get_tuner(n_features)
    tuner.search(train_dataset, validation_data=valid_dataset,
                 callbacks=[ClearTrainingOutput()])
    tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    latest_path = None
    latest_epoch = 0
    is_final = False
    for path in glob.glob(F'{MODEL_SAVE_FOLDER}/{args.model_name}/*.index'):
        if 'final' in path:
            latest_path = path[:-6]
            is_final = True
            break
        epoch = int(path[:-6].split('\\')[-1])
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path[:-6]
    if latest_path:
        print(f'loading latest ckpt: {latest_path}')
        model.load_weights(latest_path)

    if not is_final:
        model.fit(train_dataset, validation_data=valid_dataset, epochs=MAX_EPOCH,
                  callbacks=[CheckpointSave(args.model_name),
                             EarlyStopAndSave(args.model_name, patience=EARLY_STOP_PATIENCE,
                                              restore_best_weights=True)
                             ]
                  )

    return model, selected_cols, scaler


def prepare_train_datasets(args):
    np_train_list, selected_cols, scaler = load_dataset(TRAIN_FOLDER)
    np_train_list, np_valid_list = train_valid_split(np_train_list)
    train_dataset = HAIDataLoader(np_train_list,
                                  length=args.sequence_length,
                                  stride=args.sequence_stride,
                                  batch_size=args.batch_size).prefetch(10)
    valid_dataset = HAIDataLoader(np_valid_list,
                                  length=args.sequence_length,
                                  stride=args.sequence_stride,
                                  batch_size=args.batch_size).prefetch(10)
    return train_dataset, valid_dataset, selected_cols, scaler
