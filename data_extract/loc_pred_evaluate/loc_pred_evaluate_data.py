from torch.utils.data import Dataset, DataLoader


class LPEDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, data):
        data_dict = data
        # 获得用户的轨迹信息
        user_idx = data_dict.keys()
        user_list = []
        for user_id in user_idx:
            user_list.append(data_dict[user_id])
        self.user_list = user_list

    def __getitem__(self, index):
        return self.user_list[index]

    def __len__(self):
        return len(self.user_list)
