import torch
import os
from torch import nn
from mycode.GenderClassificationNN import GenderClassificationNN
from mycode.train_step import train_step
from mycode.test_step import test_step
from mycode.GenderDataset import GenderDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from typing import Tuple, Dict, List
from torchvision import transforms
from torchvision import datasets
from tqdm.auto import tqdm
from timeit import default_timer as timer 
import matplotlib.pyplot as plt

def train(model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
        epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), leave=True, color='green'):
        train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}\n"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
            "train_acc": [...],
            "test_loss": [...],
            "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.savefig('models/results.png')

def main_func():
    
    # Augment train data
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    # Don't augment test data, only reshape
    test_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    #path
    train_dir = "dataset/train"
    test_dir  = "dataset/test"
    # Using custom dataset class
    train_data = GenderDataset(targ_dir=train_dir,transform=train_transforms)
    test_data = GenderDataset(targ_dir=test_dir,transform=test_transforms)

    # OR usign the implemented default one (more efficient) -->TRY CUSTOM FIRST
    # train_data = datasets.ImageFolder(root=train_dir, # target folder of images
    #                             transform=train_transforms, # transforms to perform on data (images)
    #                             target_transform=None) # transforms to perform on labels (if necessary)

    # test_data = datasets.ImageFolder(root=test_dir, 
    #                             transform=test_transforms)

    BATCH_SIZE = 512
    #Dataloader
    train_dataloader = DataLoader(dataset=train_data, 
                                    batch_size=BATCH_SIZE, 
                                    num_workers=0, 
                                    shuffle=True) 

    test_dataloader = DataLoader(dataset=test_data, 
                                    batch_size=BATCH_SIZE, 
                                    num_workers=2, 
                                    shuffle=False) 
    
    #Set the device to run
    device= (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #define the model
    model = GenderClassificationNN().to(device)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    #training and testing
    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 150

    start_time = timer()

    model_results = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS
                    )
    end_time = timer()
    print(f"\nTotal training time: {end_time-start_time:.3f} seconds")

    # plot the results 
    plot_loss_curves(model_results)
    
    #save the image
    torch.save(model.state_dict(), "models/model.pth")  #Saving models
    print("Saved PyTorch Model State to models/model.pth")



if (__name__=='__main__'):
    main_func()
