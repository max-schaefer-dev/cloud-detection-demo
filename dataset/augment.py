import albumentations as A

def prepare_train_augmentation() -> A.Compose:
    '''Create an Albumentation Compose object with used augmentations for training'''

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    return transform

def prepare_val_augmentation():
    '''Create an Albumentation Compose object with used augmentations for validation'''

    transform = A.Compose([
    ])

    return None