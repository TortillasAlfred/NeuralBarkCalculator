from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader


def compute_mean_std(working_dir: str):
    train_dataset = RegressionDatasetFolder(working_dir,
                                            transform=Compose(
                                                [ToTensor()]))
    loader = DataLoader(train_dataset, batch_size=100)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()


def get_mean_std():
    # Util function to not have to recalculate them
    # every single time
    mean = [0.5494703054428101, 0.46148523688316345, 0.3453504145145416]
    std = [0.3568655550479889, 0.31607526540756226, 0.25801268219947815]

    return mean, std
