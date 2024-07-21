import torchvision.transforms as transforms
from PIL import Image, ImageOps

class HistogramEqualization:
    def __call__(self, img):
        return ImageOps.equalize(img)
    
# We use the recommended transformation from : https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2021.629134/full
# We noticed some lower contrast images in the dataset, so we decided to apply histogram equalization to improve the contrast.
# Define transformations for the training set
train_transforms = transforms.Compose([
    #HistogramEqualization(),  # Apply histogram equalization
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    #transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
    transforms.Normalize((0.5,), (0.5,))
])

verify_transforms_512 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define transformations for the test set (usually, we don't apply augmentation to the test set)
test_transforms_512 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

verify_transforms_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define transformations for the test set (usually, we don't apply augmentation to the test set)
test_transforms_256 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transformations = {
    'verify_transforms_512': verify_transforms_512,
    'test_transforms_512': test_transforms_512,
    'verify_transforms_256': verify_transforms_256,
    'test_transforms_256': test_transforms_256,
    'train_transforms': train_transforms
}

def get_transform(transform_name):
    return transformations[transform_name]