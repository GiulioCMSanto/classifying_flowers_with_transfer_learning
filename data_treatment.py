import torch
from torchvision import datasets, transforms
from PIL import Image

def load_transform_data(data_dir):
    """
    This function makes the necessary transformations in the training, validation and testing data.
    
    Arguments:
        data_dir: global directory of the images.
        
    Returns:
        trainloader: data for training
        validloader: data for validation
        testloader: data for testing
        train_data: data for tracking the class_to_idx transformation
    """
    
    #Make training, validation and testing images directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Create transformations for training, validation and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    
    Arguments:
        image: a jpg image
        
    Returns: processed image ready for prediction.
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #Loading image with PIL (https://pillow.readthedocs.io/en/latest/reference/Image.html)
    im = Image.open(image)
    
    #Notice: the very same transformation applied to the test/validation data will be applied here
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])
    
    return process(im)
    