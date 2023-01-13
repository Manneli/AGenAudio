import dataloader as dl
import torch
from model import audio_model
import numpy as np

NUM_CLASS = 10

def main():
    dataset = dl.AGenDataset("/home/anneli/AGenAudio/train.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True,)

    model = audio_model(num_class=NUM_CLASS,length=100)
    model = model.type(torch.float64)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    loss = torch.nn.CrossEntropyLoss()

    train(model,dataloader,optimizer,loss,num_classes=NUM_CLASS)
    #debug(model,dataloader,optimizer,loss)

# def debug(model,dataloader,optimizer,loss_function):
#     dataset = dl.AGenDataset("/home/anneli/AGenAudio/trainClass0(classes10).dataset")
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last = True)
#
#     for audio,label in dataloader:
#         audio = torch.slice_copy(audio, dim=1, start=0, end=100, step=1)
#
#         optimizer.zero_grad()
#
#         truth = make_truth(label, 10, 7, 80)
#         output = model(audio)
#
#         loss = loss_function(output, truth)
#         #print(truth)
#         print(output)
#         print(loss)
#         loss.backward()
#         optimizer.step()
#
#



def train(model,dataloader,optimizer,loss_function,epochs=5,num_classes=10):
    for ep in range(epochs):
        print("epochs: ", ep)
        for audio, label in dataloader:
            audio = torch.slice_copy(audio,dim= 1,start=0,end=100,step=1)

            optimizer.zero_grad()

            truth = make_truth(label,num_classes,7,80)
            output = model(audio)

            loss = loss_function(output,truth)
            print(truth)
            print(output)
            print(loss)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "/home/anneli/AGenAudio/model.pt")



def make_truth (label,bins,start,end):
    label = label.numpy()
    truths = np.zeros(label.shape)
    dist = (end-start)/bins
    for i in range(bins):
        truths = truths + ( (label >= ((i*dist)+start)) *  (label < (( (i+1) *dist)+start)) ) * i
    return torch.as_tensor(truths.astype(int))



if __name__ == "__main__":
    main()
