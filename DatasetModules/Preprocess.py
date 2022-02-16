import pandas as pd
import glob

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_dataset(data_folder_dir, selected_cols=None, scaler=None, ewm=True, validation=False, scaler_type='standard'):
    """
    :param data_folder_dir: directory of dataset folder. '.../folder/'
    :param selected_cols: selected columns to use. If None, automatically generates and returns
    :param scaler: fitted scaler to train_dataset. If None, automatically generates and returns
    :param ewm: whether to use Exponential smoothing.
    :return: if cols, scaler are None, gives list of np arrays. else, given with cols, scaler (train)
    """
    print(f"loading dataset...from {data_folder_dir}. scaler_type={scaler_type}")
    dirs = glob.glob(data_folder_dir + '*.csv')
    data_list = [pd.read_csv(path) for path in dirs]
    itc = [0]  # indices to cut
    if not selected_cols or not scaler:  # train
        integrated = pd.DataFrame(columns=data_list[0].columns)
        for i, data in enumerate(data_list):
            integrated = integrated.append(data)
            itc.append(integrated.shape[0])
        selected_cols = get_selected_columns(integrated)
        data_pd = integrated[selected_cols]
        assert check_isnan(data_pd), "there exist nan in dataset"
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        data_np = scaler.fit_transform(data_pd)
        if ewm:
            return [pd.DataFrame(data_np[itc[i]:itc[i+1]]).ewm(alpha=0.9).mean().values
                    for i in range(len(itc) - 1)], selected_cols, scaler
        else:
            return [data_np[itc[i]:itc[i + 1]] for i in range(len(itc) - 1)], selected_cols, scaler
    else:  # test
        integrated = pd.DataFrame(columns=selected_cols)
        for i, data in enumerate(data_list):
            integrated = integrated.append(data[selected_cols])
            itc.append(integrated.shape[0])
        assert check_isnan(integrated), "there exist nan in dataset"
        data_np = scaler.transform(integrated)
        if ewm:
            if validation:
                return [pd.DataFrame(data_np[itc[i]:itc[i + 1]]).ewm(alpha=0.9).mean().values
                        for i in range(len(itc) - 1)], data_list[0]['attack'].values
            else:
                return [pd.DataFrame(data_np[itc[i]:itc[i + 1]]).ewm(alpha=0.9).mean().values
                        for i in range(len(itc) - 1)]
        else:
            if validation:
                return [data_np[itc[i]:itc[i + 1]] for i in range(len(itc) - 1)], data_list[0]['attack'].values
            else:
                return [data_np[itc[i]:itc[i + 1]] for i in range(len(itc) - 1)]


def check_isnan(df):
    isnan = df.isna().any()
    for col in df.columns:
        if isnan[col]:
            return False
    return True


def get_selected_columns(integrated):
    selected_columns = set(integrated.columns.drop('timestamp'))
    mins, maxes = integrated.min(), integrated.max()
    for col in integrated.columns:
        if mins[col] == maxes[col]:
            selected_columns.remove(col)
    return sorted(list(selected_columns))


def train_valid_split(np_data_list, valid_ratio=0.2):
    np_train_list = []
    np_valid_list = []
    for np_data in np_data_list:
        cut_idx = int(len(np_data) * (1 - valid_ratio))
        np_train_list.append(np_data[:cut_idx, :])
        np_valid_list.append(np_data[cut_idx:, :])
    return np_train_list, np_valid_list


if __name__ == "__main__":
    train, selected_cols, scaler = load_dataset('../datasets/train/')
    test = load_dataset('../datasets/test/', selected_cols, scaler)