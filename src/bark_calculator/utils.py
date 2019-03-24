from dataset import RegressionDatasetFolder

from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader


def compute_mean_std(working_dir: str):
    train_dataset = RegressionDatasetFolder(working_dir,
                                            transform=Compose(
                                                [Resize((224, 224)), ToTensor()]))
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
    mean = [0.5495320558547974, 0.46154847741127014, 0.34539610147476196]
    std = [0.35342904925346375, 0.3120446503162384, 0.25366029143333435]

    return mean, std
