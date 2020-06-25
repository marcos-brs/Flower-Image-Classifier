import argparse
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description='Flower Classifcation predictor')
    
    parser.add_argument('--input', type=str, default='flower.jpg', help='Image Input Path')
    parser.add_argument('--checkpoint', type=str, default='my_checkpoint.pth', help='Model Checkpoint Path')
    parser.add_argument('--top_k', type=int, default=5, help='Top class')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Category Names JSON file')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or not')
    
    return parser.parse_args()

def loadmodel(file, device):
    checkpoint = torch.load(file)
    model_loaded = checkpoint['model']
    
    model_loaded = model_loaded.to(device, dtype=torch.float)

    model_loaded.class_to_idx = checkpoint['class_to_idx']
    model_loaded.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model_loaded.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model_loaded, optimizer, scheduler

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    
    crop_size = 224
    width, height = image.size
    left = (width - crop_size)/2
    top = (height - crop_size)/2
    right = (width + crop_size)/2
    bottom = (height + crop_size)/2
    image = image.crop((left, top, right, bottom))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    image = (image_array - mean) / std
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    args = get_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.to(device, dtype=torch.float)
    
    model.eval()
    
    with torch.no_grad():
        output = model(image).cpu()
        prob, idxs = torch.topk(output, topk)
        #_, preds = torch.max(output.data, 1)
    
        # convert indices to classes
        idxs = np.array(idxs)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        
        # map the class name with collected topk classes
        names = []
        for cls in classes:
            names.append(cat_to_name[str(cls)])
        
        return prob, names

def main():
    args = get_parser()
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    
    model_loaded, optimizer, scheduler = loadmodel(args.checkpoint, device)
    
    image_pil = Image.open(args.input)
    x_pos, y_pos = predict(image_pil, model_loaded, args.top_k, device)

    ax_img = imshow(process_image(image_pil))
    ax_img.set_title(y_pos[0])
    ax_img.axis('off')

    plt.figure(figsize=(4,4))
    plt.barh(range(len(y_pos)), np.exp(x_pos[0]))
    plt.yticks(range(len(y_pos)), y_pos)

    plt.show()

if __name__ == "__main__":
    main()