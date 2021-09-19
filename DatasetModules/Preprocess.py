import pandas as pd
import glob

from sklearn.preprocessing import StandardScaler


def load_dataset(data_folder_dir, selected_cols=None, scaler=None, ewm=True, validation=False):
    """
    :param data_folder_dir: directory of dataset folder. '.../folder/'
    :param selected_cols: selected columns to use. If None, automatically generates and returns
    :param scaler: fitted scaler to train_dataset. If None, automatically generates and returns
    :param ewm: whether to use Exponential smoothing.
    :return: if cols, scaler are None, gives list of np arrays. else, given with cols, scaler (train)
    """
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
        scaler = StandardScaler()
        data_np = scaler.fit_transform(data_pd)
        if ewm:
            return [pd.DataFrame(data_np[itc[i]:itc[i+1]]).ewm(alpha=0.9).mean().values
                    for i in range(len(itc) - 1)], selected_cols, scaler
        else:
            return [data_np[itc[i]:itc[i + 1]] for i in range(len(itc) - 1)], selected_cols, scaler
    else:  # test
        if validation:
            attack = data_list[0]['attack']
        integrated = pd.DataFrame(columns=selected_cols)
        for i, data in enumerate(data_list):
            integrated = integrated.append(data[selected_cols])
            itc.append(integrated.shape[0])
        assert check_isnan(integrated), "there exist nan in dataset"
        data_np = scaler.transform(integrated)
        if ewm:
            if validation:
                return [pd.DataFrame(data_np[itc[i]:itc[i + 1]]).ewm(alpha=0.9).mean().values
                        for i in range(len(itc) - 1)], attack
            else:
                return [pd.DataFrame(data_np[itc[i]:itc[i + 1]]).ewm(alpha=0.9).mean().values
                        for i in range(len(itc) - 1)]
        else:
            if validation:
                return [data_np[itc[i]:itc[i + 1]] for i in range(len(itc) - 1)], attack
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