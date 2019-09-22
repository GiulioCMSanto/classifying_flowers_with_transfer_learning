#Import Libraries used for training
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models
from data_treatment import load_transform_data
from classifiers import Network

def load_model(arch = 'vgg16'):
    """
    This function loads a pre-trained model with a given architecture.
    
    Arguments:
        arch: the desired architecture (vgg11 vgg13, vgg16, vgg19).
        
    Returns: pre-trained model
    """
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.name = 'vgg11'
        return model
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = 'vgg13'
        return model
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
        return model
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.name = 'vgg19'
        return model
    else:
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
        return model
    
def build_classifier(model,
                     hidden_layers = [2048, 1024]):
    """
    This function builds a classifier for the flower classification problem.
    The classifier is added to the pre-trained model.
    
    Arguments:
        model: the pre-trained model.
        hidden_layers: the number of desired inputs for both hidden_layers.
        
    Returns: a full-structured model ready for training.
    """
    
    #Add default hidden layer in case a None is passed through the terminal
    if not hidden_layers:
        hidden_layers = [2048, 1024]
    
    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #Create new classifier with 102 output flower classes
    classifier = Network(input_size = 25088,
                         output_size = 102,
                         hidden_layers = hidden_layers,
                         drop_p = 0.2)

    #Update the original classifier
    model.classifier = classifier
   
def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    """
    This function trains a fully-structured model.
    
    Arguments:
        model: the full model
        trainloader: data in loader form for training
        validloader: data in loader form for validation during training
        epochs: number of epochs for training
        print_every: interval of execution loop for printing log
        criterion: loss criterion (cost function)
        optimizer: optimizer used for updating the network weights
        device: cpu or gpu execution
    """
    
    print(f'Starting Model Training on {device}... \n')
    epochs = epochs
    print_every = print_every
    
    model.to(device)
    model.train()
    
    running_loss = 0
    steps = 0
    
    for epoch in range(epochs):
        
        for images, labels in trainloader:
            
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:

                        images, labels = images.to(device), labels.to(device)

                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        # Calculate Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
    print("Model Sucefully Trained! \n")
    
def network_validation(model, testloader, criterion, device='cpu'):
    """
    This function tests the trained network.
    
    Arguments:
        model: the trained model
        testloader: data in loader format for model testing (validation)
        criterion: the loss function
        device: cpu or gpu
        
    Returns: print testing accuracy
    """
    print('Starting Validation in the Testing Set...\n')
    model.to(device)
    model.eval()
    test_loss = 0
    steps = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in testloader:
            steps+=1
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()
            
            #Calculate Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Network Test Accuracy: {accuracy/steps} \n")
    
def save_model(model, data, filepath):
    """
    This function saves the trained model.
    
    Arguments:
        model: trained model
        data: the data used in order to extract class_to_idx transformation
        filepath: path to save the model
    """
    
    print("Saving Model ... \n") 
    #Extract the class_to_idx transformation 
    model.class_to_idx = train_data.class_to_idx
    #Put the model in CPU mode to allow predictions without having CUDA
    model.to('cpu')

    #Create the checkpoint
    checkpoint = {'input_size':25088,
                  'output_size':102,
                  'hidden_layers':[each.out_features for each in model.classifier.hidden_layers],
                  'drop_p':0.2,
                  'state_dict':model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model_type':model.name}
    
    #Save the model
    try:
        torch.save(checkpoint, filepath)
        print("Model Sucefully Saved")
    except:
        print("Some error ocurred while saving the model")
    
    
          
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a Deep Neural Network for Flowers Classification')
    parser.add_argument('data_dir', help='Directory of the Images')
    parser.add_argument('-s', '--save_dir', metavar='', help='Directory to Save the Trained Model')
    parser.add_argument('-a', '--arch', metavar='', help='Pre-trained Model Architecture')
    parser.add_argument('-l', '--learning_rate', metavar='', type=float, help='Learning Rate for Training Model')
    parser.add_argument('-H', '--hidden_units', metavar='', nargs=2, type=int, help='Number of Inputs from each Hidden Unit')
    parser.add_argument('-e', '--epochs', metavar='', type=int, help='Number of Epochs')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for Training')
    args = parser.parse_args()
    
    #Load data and make necessary transformations
    trainloader, validloader, testloader, train_data = load_transform_data(data_dir = args.data_dir)
    
    #Load model if desired architecture
    model = load_model(arch = args.arch)
    
    #Build classifier and incorporate it into the pre-trained model
    build_classifier(model = model,
                     hidden_layers = args.hidden_units)
    
    #Define the loss function (criterion), the optimizer and the device
    criterion = nn.NLLLoss()
    
    #Define Learning Rate
    if not args.learning_rate:
        lr = 0.001
    else:
        lr = args.learning_rate
    
    #Build optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    #Define device
    if not args.gpu:
        device = 'cpu'
    else:
        device = 'cuda'
    
    #Train the model
    if not args.epochs:
        epochs = 4
    else:
        epochs = args.epochs
    
    train_model(model = model,
                trainloader = trainloader,
                validloader = validloader,
                epochs = epochs,
                print_every = 25,
                criterion = criterion,
                optimizer = optimizer,
                device = device)
    
    #Test the Model
    network_validation(model=model,
                       testloader=testloader,
                       criterion=criterion,
                       device=device)
    
    #Save the model
    if not args.save_dir:
        filepath = 'checkpoint.pth'
    else:
        filepath = args.save_dir + 'checkpoint.pth'
    
    save_model(model=model, data=train_data, filepath=filepath)
          