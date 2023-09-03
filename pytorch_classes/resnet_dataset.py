import torch


class ResNetDataset(torch.utils.data.Dataset):
    """Rubik's Cube dataset."""
    def __init__(self, color_list, label_list):
        self.color_list = torch.Tensor(color_list).double()
        print(self.color_list.shape)
        self.label_list = torch.Tensor(label_list).double()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        X = self.color_list[index]
        y = self.label_list[index]
        return X, y