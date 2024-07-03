import torch
from tqdm import tqdm
from tqdm.rich import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize 


def main():
    data_transform = Compose([ToTensor(), Resize((128, 128))])
    root = "imagenette2-160/train"

    train_imgs = ImageFolder(root=root, transform=data_transform)
    val_imgs = ImageFolder(root=root, transform=data_transform)

    img_shape = train_imgs[0][0].shape

    # load all the images into memory and store them as a tensor
    train_img_tensor = torch.zeros((len(train_imgs), *img_shape))
    train_labels_tensor = torch.zeros(len(train_imgs))

    for i in tqdm(range(len(train_imgs)), desc="Loading train images"):
        train_img_tensor[i] = train_imgs[i][0]
        train_labels_tensor[i] = train_imgs[i][1]
    
    torch.save(
        {"images": train_img_tensor, "labels": train_labels_tensor},
        "imagenette2-128x128-train.pt"
    )

    val_img_tensor = torch.zeros((len(val_imgs), *img_shape))
    val_labels_tensor = torch.zeros(len(val_imgs))
    for i in tqdm(range(len(val_imgs)), desc="Loading val images"):
        val_img_tensor[i] = val_imgs[i][0]
        val_labels_tensor[i] = val_imgs[i][1]

    torch.save(
        {"images": val_img_tensor, "labels": val_labels_tensor},
        "imagenette2-128x128-val.pt"
    )


if __name__ == "__main__":
    main()