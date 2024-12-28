''' 
1 data - transformation, datasets, and dataloaders 
'''

def dataloader(train_dir, valid_dir, test_dir, augment=True, img_size=224, batch_size=64):
    ''' Return data loaders for training, validation, and testing '''
    import torch
    from torchvision.transforms import v2
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    transform_no_aug = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_aug = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=30),
        v2.RandomResizedCrop(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  

    train_dataset = ImageFolder(root=train_dir, transform=transform_aug if augment else transform_no_aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = ImageFolder(root=valid_dir, transform=transform_no_aug)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    test_dataset = ImageFolder(root=test_dir, transform=transform_no_aug)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, valid_loader, test_loader








