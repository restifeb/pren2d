import os
import cv2
from data.data_utils import Augmenter
import random

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class listDataset(Dataset):
    def __init__(self, source=None, list_file=None, transform=None, inTrain=False, p_aug=0, vert_test=False):
        '''
        :param imgdir: path to root directory
        :param list_file: path to ground truth file
        :param transform: torchvison transforms object
        :param inTrain: True for training stage and False otherwise
        :param p_aug: probability of data augmentation
        '''

        list_file = list_file
        # for filename in self.list_file:
        #     file_path = os.path.join(imgdir, filename)


        #     print(file_path)
        #     # if not os.path.exists(filename):
        #     #     raise FileNotFoundError('File not found: {}'.format(filename))
                
        #     with open(os.path.join(file_path, 'train.tsv')) as fp:
        #         self.lines = fp.readlines()
        #         self.nSamples = len(self.lines)

        self.transform = transform
        self.source = source
        self.inTrain = inTrain
        self.p_aug = p_aug
        self.vert_test = vert_test
        self.data_lines = self.get_image_info_list(list_file)
        self.nSamples = len(self.data_lines)

        if inTrain:
            self.aug = Augmenter(p=self.p_aug)

    def get_image_info_list(self, list_file):
        if isinstance(list_file, str):
            list_file = [list_file]
        data_lines = []
        for idx, file in enumerate(list_file):
            dataset_source = os.path.join(self.source, file)
            if not os.path.exists(dataset_source):
                raise FileNotFoundError(f"{file} not found")

            if not os.path.isdir(dataset_source):
                raise NotADirectoryError(f"{file} is not a directory")

            if os.path.exists(os.path.join(dataset_source, "img")):
                folder_path = "img"

            elif os.path.exists(os.path.join(dataset_source, "data")):
                folder_path = "data"

            folder_path = f"{file}/{folder_path}/"
            split = ''
            # if self.mode == "eval":
            #     split = 'valid'
            # else:
            #     split = self.mode
            labels_path = os.path.join(dataset_source, "train.tsv")
            with open(labels_path, "r") as f:
                lines = f.readlines()
                lines = [folder_path + line.split('\t', 0)[0] for line in lines[1:] if line.strip()]  # Ignore the first row and empty lines
                data_lines.extend(lines)
        return data_lines

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # -- get image
        data_line = self.data_lines[index]
        line_splits = data_line.split("\t")
        imgpath = os.path.join(self.source, line_splits[0])

        img = cv2.imread(imgpath)

        # ignore invalid images
        if img is None:
            #print('Invalid image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # ignore too small images
        h, w, _ = img.shape
        if min(h, w) <= 5:
            # print('Too small image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- get text label
        label = ' '.join(line_splits[1:])
        label = label.lower()

        # ignore too long texts in training stage
        if len(label) >= 25 and self.inTrain:
            # print('Too long text: {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- data preprocess
        if self.inTrain:
            img = self.aug.apply(img, len(label))

        x = self.transform(img)
        x.sub_(0.5).div_(0.5)  # normalize to [-1, 1)

        # for vertical test samples, return rotated versions
        x_clock, x_counter = 0, 0
        is_vert = False
        if self.vert_test and not self.inTrain and h > w:
            is_vert = True
            img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_clock = self.transform(img_clock)
            x_counter = self.transform(img_counter)
            x_clock.sub_(0.5).div_(0.5)
            x_counter.sub_(0.5).div_(0.5)

        return (x, label, x_clock, x_counter, is_vert, imgpath)


def TrainLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])

    dataset = listDataset(source=configs.image_dir,
                          list_file=configs.train_list,
                          transform=transform,
                          inTrain=True,
                          p_aug=configs.aug_prob)

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=True,
                      num_workers=configs.workers)


def TestLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])

    dataset = listDataset(source=configs.image_dir,
                          list_file=configs.val_list,
                          transform=transform,
                          inTrain=False,
                          vert_test=configs.vert_test)

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=False,
                      num_workers=configs.workers)


if __name__== '__main__':

    from Configs.trainConf import configs
    import matplotlib.pyplot as plt

    train_loader = TrainLoader(configs)
    l = iter(train_loader)
    im, la, *_ = next(l)
    for i in range(100):
        plt.imshow(im[i].permute(1,2,0) * 0.5 + 0.5)
        plt.show()

    # import matplotlib.pyplot as plt
    # from Configs.testConf import configs
    # valloader = TestLoader(configs)
    # l = iter(valloader)
    # im, la, *_ = next(l)
    # plt.imshow(im[0].permute(1, 2, 0) * 0.5 + 0.5)

