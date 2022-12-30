import dataloader as dl
import torch
from model import audio_model
import numpy as np

def main():
    dataset = dl.AGenDataset("/home/anneli/AGenAudio/train.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    model = audio_model()
    model = model.type(torch.float64)

    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()

    train(model,dataloader,optimizer,loss)


def train(model,dataloader,optimizer,loss_function,epochs=5,num_classes=10):
    for ep in range(epochs):
        print("epochs: ", ep)
        for audio, label in dataloader:
            optimizer.zero_grad()

            truth = make_truth(label,num_classes,7,80)
            output = model(audio)

            print(label)

            print(truth)
            print(output)

            loss = loss_function(output,truth)
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
