from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True)
