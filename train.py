import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms

from losses import FocalLoss, mIoULoss
from model import UNet
from dataloader import segDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Semantic segmentation dataset', help='path to your dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='dnumber of epochs')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    N_EPOCHS = args.num_epochs
    BACH_SIZE = args.batch

    color_shift = transforms.ColorJitter(.1,.1,.1,.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(args.data, training = True, transform= t)

    print('Number of data : '+ str(len(dataset)))

    test_num = int(0.1 * len(dataset))
    print(f'test data : {test_num}')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num], generator=torch.Generator().manual_seed(101))
    N_DATA, N_TEST = len(train_dataset), len(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=6).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')


    model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))

    os.makedirs('./saved_models', exist_ok=True)

    plot_losses = []
    scheduler_counter = 0

    for epoch in range(N_EPOCHS):
        # training
        model.train()
        loss_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):

            pred_mask = model(x.to(device))  
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    N_EPOCHS,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1
        # testing
        model.eval()
        val_loss_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():    
                pred_mask = model(x.to(device))  
            val_loss = criterion(pred_mask, y.to(device))

            val_loss_list.append(val_loss.cpu().detach().numpy())
            
        print(' epoch {} - loss : {:.5f} - val loss : {:.5f}'.format(epoch, 
                                                            np.mean(loss_list), 
                                                            np.mean(val_loss_list)))
        plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best == True:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(), './saved_models/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
        
        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0


    # plot loss
    plot_losses = np.array(plot_losses)
    plt.figure(figsize=(12,8))
    plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
    plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
    plt.title(args.loss, fontsize=20)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('loss',fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation']) # using a named size
    plt.savefig('loss_plots.png')

