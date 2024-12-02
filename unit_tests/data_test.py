import unittest
from utils.datasets import get_dataloaders

class TestData(unittest.TestCase):

    def setUp(self):
        # init the paths and batch size
        self.ncars_train_dataset_path = "../data/ncars/ave_32x32_DATASETS/plain/train_n_cars_dataset_poolingave_1framepereventset_plain.pth"
        self.ncars_valid_dataset_path = "../data/ncars/ave_32x32_DATASETS/plain/valid_n_cars_dataset_poolingave_1framepereventset_plain.pth"
        self.ncars_test_dataset_path = "../data/ncars/ave_32x32_DATASETS/plain/test_n_cars_dataset_poolingave_1framepereventset_plain.pth"
        self.batch_size = 32

        self.nmnist_train_dataset_path = "../data/nmnist/Plain/Plain_1FramePerEventSet_train_dataset.pth"
        self.nmnist_valid_dataset_path = "../data/nmnist/Plain/Plain_1FramePerEventSet_valid_dataset.pth"
        self.nmnist_test_dataset_path = "../data/nmnist/Plain/Plain_1FramePerEventSet_test_dataset.pth"

    def test_dataloaders_ncars(self):
        train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
            self.ncars_train_dataset_path,
            self.ncars_valid_dataset_path,
            self.ncars_test_dataset_path,
            self.batch_size
        )

        # check the size of the datasets
        self.assertEqual(len(train_dataloader.dataset), 11566)
        self.assertEqual(len(valid_dataloader.dataset), 3856)
        self.assertEqual(len(test_dataloader.dataset), 8607)

    def test_dataloaders_nmnist(self):
        train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
            self.nmnist_train_dataset_path,
            self.nmnist_valid_dataset_path,
            self.nmnist_test_dataset_path,
            self.batch_size
        )

        self.assertEqual(len(train_dataloader.dataset), 50000)
        self.assertEqual(len(valid_dataloader.dataset), 10000)
        self.assertEqual(len(test_dataloader.dataset), 10000)


if __name__ == "__main__":
    unittest.main()