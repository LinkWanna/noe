import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

fashion_minist_class_list = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
fashion_minist_class_encoding = {class_name: idx for idx, class_name in enumerate(fashion_minist_class_list)}


def load_fashion_mnist(root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.2860], std=[0.3530]),
        ]
    )

    train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
