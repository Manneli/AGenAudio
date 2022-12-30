import torch.utils
import numpy as np

def main():
    dataset = AGenDataset("/home/anneli/AGenAudio/validation.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,shuffle=True)

    for audio, label in dataloader:
        print(audio.size())
        print(label)

    dataset = AGenDataset("/home/anneli/AGenAudio/test.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for audio, label in dataloader:
        print(audio.shape)
        print(label)

    dataset = AGenDataset("/home/anneli/AGenAudio/train.dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for audio, label in dataloader:
        print(audio.shape)
        print(label)

class AGenDataset (torch.utils.data.Dataset) :

    def __init__(self, dataset_path, max_length = 1132):
        self.max_length = max_length
        file = open(dataset_path, "r")
        line = file.readline()
        self.data = []
        while line != "" :
            line = line.split(" ")
            self.data = self.data + [ [line[0],line[1],line[2],line[3][:-1]] ]
            line = file.readline()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]

        audio = np.load(data[0])
        audio = np.pad(audio,  ((0,0),(0,self.max_length-audio.shape[1])) )
        audio = np.transpose(audio)

        return torch.as_tensor(audio), torch.as_tensor(int(data[2]))





if __name__ == "__main__":
    main()