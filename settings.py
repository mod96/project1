TRAIN_FOLDER = 'datasets/train/'
VALIDATION_FOLDER = 'datasets/validation/'
TEST_FOLDER = 'datasets/test/'
SUBMISSION_PATH = 'datasets/sample_submission.csv'

MODEL_SAVE_FOLDER = 'saved_models/'

MAX_EPOCH = 20 # 200
SEARCH_SAVE_DIRECTORY = 'search_saved/'

EARLY_STOP_PATIENCE = 5 # 20


"""
** ABOUT KERAS-TUNER **
class 사용법 : https://ichi.pro/ko/keras-tuner-mich-hiplot-eul-sayonghan-singyeongmang-haipeo-palamiteo-tyuning-129980331205114
함수형 : https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko
kt 가 알아서 저장을 해주는듯? => kt.Hyperband의 'directory'를 건드려주면 되는 것 같은데?
                        => model을 class로 구현해버리고 builder에서는 epoch만 받고, directory는 settings.py에서.
                        => callback은 만들긴 해야하는듯 (search해서 best_hps 찾고 그거 가져와서 다시 fit하기 때문)
"""
