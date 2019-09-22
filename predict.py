import argparse
import torch
import json
from classifiers import Network
from data_treatment import process_image
from train import load_model
import sys

def load_checkpoint(filepath):
    """
    This function Loads the trained model.
    
    Arguments:
        filepath: the path were the model is saved.
        
    Returns: trained model
    """
    
    print(f"Loading Trained Model ... \n")
    
    #Load checkpoint
    checkpoint = torch.load(filepath)    
    
    #Load pre-trained model with same architecture used for training
    model = load_model(arch = checkpoint['model_type'])
    print(f"Model of Type {model.name} Sucefully Loaded \n")
    
    #Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #Rebuild Classifier with same structure used for training
    classifier = Network(input_size = checkpoint['input_size'],
                         output_size = checkpoint['output_size'],
                         hidden_layers = checkpoint['hidden_layers'],
                         drop_p = checkpoint['drop_p'])
    
    #Update classifier with training results (weights and class/idx transformation)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict']) #That actually loads the state dictionary
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, device, top_k, category_names=None):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path: the global path for the training, testing and validation images
        model: the trained model
        device: cpu or gpu
        top_k: the desired number of most probable predictions
        category_names: json file for converting class numbers into names
        
    Returns:
        top_p: probabilities of top_k classes
        top_names: top_k classes names
        pred_classes: top_k classes numbers  
    '''
    #Process the image using the function process_image created
    im = process_image(image_path)
    im.unsqueeze_(0)
    try:
        model.to(device).float() #Put the model in the choosen device
    except AssertionError as e:
        print(e)
        sys.exit(1)
    
    #Put the model into evaluation mode
    model.eval()
    
    #Make the prediction (forward pass)
    with torch.no_grad():
        pred = model.forward(im.to(device))
    
    #Take the exponential of the output (which is in log_softmax)
    ps = torch.exp(pred)
    
    #Take top-k classifications
    top_p, top_class = ps.topk(top_k, dim=1)
    top_p = top_p.tolist()
    
    #Convert Index To Class
    classes = top_class.tolist()[0] #Class indexes

    pred_classes = []
    for value in classes:
        pred_classes.append(list(model.class_to_idx)[value]) #Take the class from the index
            
    #Convert class index to class value
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        #Obtain the corrisponding class name using cat_to_name json file
        top_names = []
        for pred_class in pred_classes:
            top_names.append(cat_to_name[pred_class])
    else:
        top_names = None
        
    return top_p, top_names, pred_classes

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict the Class of a Given Flower')
    parser.add_argument('image_path', help='Path to the Image Being Predicted path/to/image')
    parser.add_argument('checkpoint', help='Trained Model Checkpoint Path')
    parser.add_argument('-k', '--top_k', metavar='', type = int, help='K Most Probable Classes')
    parser.add_argument('-n', '--category_names', metavar='', help='Explicit Predicted Flowers Classes Names (json input)')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for Prediction')
    args = parser.parse_args()
    
    #Load Model
    model = load_checkpoint(args.checkpoint)
    
    #Setup device
    if not args.gpu:
        device = 'cpu'
    else:
        device = 'cuda'
    
    #Setup image path
    image_path = args.image_path
    
    #Make Prediction
    if not args.top_k:
        top_k = 1
    else:
        top_k = args.top_k 
        
    top_p, top_names, top_class = predict(image_path=image_path,
                                          model=model,
                                          device=device,
                                          top_k=top_k,
                                          category_names = args.category_names)
    
    if not args.category_names:
        print(f"Prediction: Flower of Class {top_class} with {top_p} probability")
    else:
        print(f"Prediction: Flower of Class {top_names} with {top_p} probability")
        
    
    
    
    
    