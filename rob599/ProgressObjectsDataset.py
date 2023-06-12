import os
import pickle
from typing import Any, Callable, Optional, Tuple

import random

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class ProgressObjectsDataset(VisionDataset):

    base_folder = "Progress-Objects-Sample"
    url = "https://drive.google.com/file/d/1C8_JFsnPVm392C-S1rH0y4HFfNkdMlXi/view?usp=share_link"
    filename = "Progress-Objects-Sample.tar.gz"
    tgz_md5 = "b32c587684bb54a9f918b6b081a18e28"

    object_list = [
        ["meta.pkl", "a4f9e193931f800be07633052f8e6741"],
        ["cracker_box_1.pkl", "ee621e08f1d4242d12a286c34a3ae551"],
        ["cracker_box_2.pkl", "93c7188b85347bb40c41d5501e1e471c"],
        ["cracker_box_3.pkl", "feeefc6db6bf8597fad8d010d99d4a6c"],
        ["cracker_box_4.pkl", "3712f842d358bafe39db8bd7a1b3d373"],
        ["cracker_box_5.pkl", "4bf7df78edb3e3ecbd0e12443c004e2d"],
        ["cracker_box_6.pkl", "f71b4c82fe70681a642b2f430c4e9e90"],
        ["cracker_box_7.pkl", "483b4687e35206a8d25e469117da6166"],
        ["cracker_box_8.pkl", "453847af9f9c3755984fd3e3e9bdda73"],
        ["cracker_box_9.pkl", "d136555a36d39d07ae952d792499fcf5"],
        ["cracker_box_10.pkl", "5614ec3aea2cdb97be6852f2835269fe"],
        ["cracker_box_11.pkl", "b834be803ce0367893303d3348937f68"],
        ["cracker_box_12.pkl", "77b098d9f3262f95f241e306db7cc696"],
        ["cracker_box_13.pkl", "4e0a7d2a035aee6ddcddc3422f3e7ef8"],
        ["cracker_box_15.pkl", "f2fd13c87183350bc2e4239bbd65bedf"],
        ["cracker_box_16.pkl", "6629b4a80e98a58ee1fc6a168d249329"],
        ["gelatin_box_1.pkl", "3400d4781753c41bff3aea599795ac82"],
        ["gelatin_box_2.pkl", "6a002af3e3b3918af54f4e266c81fd40"],
        ["gelatin_box_3.pkl", "d8eee7338192c4cd8b4faaba0c411f16"],
        ["gelatin_box_4.pkl", "da2bd88410d42adc0c6d24bec696a7f5"],
        ["gelatin_box_5.pkl", "838312b94391606d86161e162b456868"],
        ["gelatin_box_6.pkl", "5edbd4b72138eac4b160c806cc8d6d30"],
        ["gelatin_box_7.pkl", "2899e5da4d9f60565c80b2152d33b371"],
        ["gelatin_box_8.pkl", "f926e22d506ac2f5ffa9f1be64d5081a"],
        ["gelatin_box_9.pkl", "0b7ddec9f73db952612b0fcdae335f9b"],
        ["gelatin_box_10.pkl", "26f7396a9eb5772111d210650725b191"],
        ["gelatin_box_11.pkl", "d4ac1c78b8b4a20e71c4e9d58c05dcca"],
        ["gelatin_box_12.pkl", "9e2870309b29c07e4eedeef002fcfe7a"],
        ["gelatin_box_13.pkl", "d1f320e7acf54a94c4a683c757c70352"],
        ["gelatin_box_14.pkl", "28b6a6f209a54e42e11615791e743145"],
        ["gelatin_box_15.pkl", "2e68068601c3128f5c58ada9249fa05c"],
        ["gelatin_box_16.pkl", "51e5b761a9a737cfcc6df3a5e0d84101"],
        ["large_marker_1.pkl", "7876c96a9f02cf1c52e731ee4ea3389f"],
        ["large_marker_2.pkl", "e4416f98cd9769a72e3a6cada7c49d97"],
        ["large_marker_3.pkl", "8911af04f2015c8b0a56ceaa58a565ef"],
        ["large_marker_4.pkl", "da0b20a997e04eab28b49e50d75ed957"],
        ["large_marker_5.pkl", "50b8510349718b84eca2469909bce6e4"],
        ["large_marker_6.pkl", "97bb232f72982314cb7372dfa30b3e3c"],
        ["large_marker_7.pkl", "2b95679d0023eef37001df0314424b87"],
        ["large_marker_8.pkl", "e31399285ee80fe872ba2f457f5119c3"],
        ["large_marker_9.pkl", "7be0ca775a4b3027c1d785bdc7e27151"],
        ["large_marker_11.pkl", "8d1b79494b6af54bb49ce28d4bb71222"],
        ["large_marker_12.pkl", "bf7449f2825207150472c802a4fa778e"],
        ["large_marker_13.pkl", "3042c6f7cf5010c126decb2b8749167f"],
        ["large_marker_14.pkl", "62dfcabf98151cda649b3cdc2174619e"],
        ["large_marker_15.pkl", "58d6e588b702ff0aba85462501880a90"],
        ["large_marker_16.pkl", "6198c94eee659dd0cf3e82a4eff59eef"],
        ["master_chef_can_1.pkl", "fea190007b5132bda0922d4a1417ff78"],
        ["master_chef_can_2.pkl", "9ddf1b3efa5cb9127bdbaffe8fe2546a"],
        ["master_chef_can_3.pkl", "151248280badccb2b51cccb16c359d82"],
        ["master_chef_can_4.pkl", "c7ff431f6790e7a05ada4bea9417ed13"],
        ["master_chef_can_5.pkl", "1deb62c1122fc6956c84e63f41b32f6c"],
        ["master_chef_can_7.pkl", "410b5f14123e3db1a19f0b5d0cabcede"],
        ["master_chef_can_8.pkl", "18a688fef0ad5ad5117b36ac661dfd83"],
        ["master_chef_can_9.pkl", "b1025276c0441d7381e5d2dde2f40a6b"],
        ["master_chef_can_10.pkl", "da92543a9affd4219839d9f5093a7586"],
        ["master_chef_can_11.pkl", "aef4797b4e29b2882d076e7ebed516b6"],
        ["master_chef_can_12.pkl", "26b9e2b4879e2245e8eeb6374416a004"],
        ["master_chef_can_13.pkl", "b9567774ece1c18a04879d5b1238187e"],
        ["master_chef_can_14.pkl", "354fa32cc7975371231923a52a118c5c"],
        ["master_chef_can_15.pkl", "7c552cdfe0bc02667985997f22f62624"],
        ["master_chef_can_16.pkl", "d8f344fbf4cc52e6415abc7a3abc1828"],
        ["mug_1.pkl", "4d3b5cfd4ed26c19b600da40a672e06f"],
        ["mug_2.pkl", "b01006cf6e6b7a813ba27f6e86ed7b25"],
        ["mug_3.pkl", "128b002b657cac3dc529fb03eddf4d73"],
        ["mug_4.pkl", "8af4982032e5387d7c183f48f87d43c2"],
        ["mug_5.pkl", "e6b6e81fffa515bbd48a0f1fbbc3361c"],
        ["mug_6.pkl", "c803429764010d36a51c5ce408237290"],
        ["mug_7.pkl", "de37207c7dfbc2e2452d1f30fbe43079"],
        ["mug_8.pkl", "8b60abed6b19f69ddaeee52e0289b4f8"],
        ["mug_9.pkl", "2694c2b7d32c1d83494c2a1cbce88b9a"],
        ["mug_10.pkl", "b9615d06449d83ef2a7ea7a25b2e8ec5"],
        ["mug_11.pkl", "cb1c87c272e5cbb3cec8128808b218bc"],
        ["mug_12.pkl", "a7e0b98485cdc56a806540d484ec5bb2"],
        ["mug_13.pkl", "abfe49c97b97c677c0238e82f1d50d4f"],
        ["mug_14.pkl", "f83d700f547c722c5f8b2366c3ba951d"],
        ["mug_15.pkl", "50229176324f0c01812515fcbc0edde1"],
        ["mug_16.pkl", "1b98c2c0aa278b6eb6198a373cd994ca"],
        ["mustard_bottle_1.pkl", "a98a699f5884d9d4f7814dc3bcab964f"],
        ["mustard_bottle_2.pkl", "0d6e90e1699855250012193b416460da"],
        ["mustard_bottle_3.pkl", "b2b559538d0d878782862044330b6c52"],
        ["mustard_bottle_4.pkl", "c194f4d3516d389c92836b0d522752e2"],
        ["mustard_bottle_5.pkl", "83ff3eecf7b4f8e057cbd110e88bae9a"],
        ["mustard_bottle_6.pkl", "eb15761b7b2b8934d923272dc888b9df"],
        ["mustard_bottle_7.pkl", "899c7259e662a1f84329dd45ccd4e214"],
        ["mustard_bottle_8.pkl", "d1a90d7b457de4bd7c578b7798c5159b"],
        ["mustard_bottle_9.pkl", "f3360de450f3b59afdfb68a3b3196b31"],
        ["mustard_bottle_10.pkl", "f7a20d2e1400d26b83ccad067a45e394"],
        ["mustard_bottle_11.pkl", "83e0bad68bf21ba0f2460b6321ef62f0"],
        ["mustard_bottle_12.pkl", "7b0076bf02d6692640299d3993c2858b"],
        ["mustard_bottle_13.pkl", "6115dff9aa50748c5deef5b9b73add1d"],
        ["mustard_bottle_14.pkl", "4a7d956a021cbc681e174a355ba53b5f"],
        ["mustard_bottle_15.pkl", "9e7a61d3a6135bda03242ccb087ad460"],
        ["mustard_bottle_16.pkl", "989dbfe8579bce1ee77f899826011f23"],
        ["potted_meat_can_1.pkl", "313b2734bcd4ed0b6e92d96948fb509c"],
        ["potted_meat_can_2.pkl", "028117c702d49f0f50685b9a3cd89882"],
        ["potted_meat_can_3.pkl", "ac352dd3fbdea57add3284fa8a83298b"],
        ["potted_meat_can_4.pkl", "89a7f85b7f1ff4dd8e4b2052804ee504"],
        ["potted_meat_can_5.pkl", "cd59d247c9a63d73fb3e4b6246650f09"],
        ["potted_meat_can_6.pkl", "2f16f8b64c07b76d5e0108e9d6e692c0"],
        ["potted_meat_can_7.pkl", "60c323d74a259d94a7b4ed3978597dee"],
        ["potted_meat_can_8.pkl", "9bbf50076097b8c6ed863714fb65e0a4"],
        ["potted_meat_can_9.pkl", "c3cb73eee5a647531e7ca7c7b0bfda55"],
        ["potted_meat_can_10.pkl", "d93e1572e72183baa2a5f8145a615266"],
        ["potted_meat_can_11.pkl", "9ef2d9bb4bc62840b4218f9a088028cc"],
        ["potted_meat_can_12.pkl", "5f70d6de1576d5602c65e0668eb5636f"],
        ["potted_meat_can_13.pkl", "a05835fe95158d105682d813260fbee0"],
        ["potted_meat_can_14.pkl", "5ac75e2d90d22db821d6bb1ad2d16926"],
        ["potted_meat_can_15.pkl", "572e89afd8879e8f9e51a027cb555a4d"],
        ["potted_meat_can_16.pkl", "691655acf392091be821a397437260d8"],
        ["sugar_box_1.pkl", "0ed087578e99d726a33ff020bfadb6cc"],
        ["sugar_box_2.pkl", "873c2783962f656f45cace8357ef2245"],
        ["sugar_box_3.pkl", "61ace4b49212e7e3f1a0a04812424566"],
        ["sugar_box_4.pkl", "5d2e75d0093eafbe204adec88fd81ec6"],
        ["sugar_box_5.pkl", "88822b4df28cbf7904b43ebe3b924de5"],
        ["sugar_box_6.pkl", "780035ad27c6dc286ab21bdc39ae643d"],
        ["sugar_box_7.pkl", "c0257687de84a1a0edea596e0ddfa501"],
        ["sugar_box_8.pkl", "b0c68a1cce7b35651c3141549c24e75f"],
        ["sugar_box_9.pkl", "39155dba336b99fb50fa6f607e60582c"],
        ["sugar_box_10.pkl", "66425f0a8766f7e48f14455c8da9b5fd"],
        ["sugar_box_11.pkl", "6df2893cd4b808ac602f05fe4f1f8335"],
        ["sugar_box_12.pkl", "533ed7feb9afba9b42d5b085bea7ccd6"],
        ["sugar_box_13.pkl", "26d6d9e549119eb38bf0e43f13798de7"],
        ["sugar_box_14.pkl", "86d34de5cf091d108d39bc3cb33d7f7d"],
        ["sugar_box_15.pkl", "46ef4407017f4b060cca4693f7c919b7"],
        ["sugar_box_16.pkl", "452e30866c58661726db30ae1a5e62b9"],
        ["tomato_soup_can_1.pkl", "14f3cce9c66a4c1c9c69c1a9ab677482"],
        ["tomato_soup_can_2.pkl", "90e820dc0a330c3566a847cae7a3b8e4"],
        ["tomato_soup_can_3.pkl", "e28c812ee773395d12b1f755950be0ba"],
        ["tomato_soup_can_4.pkl", "41ed4c37831c2885f4f1b6c37f837d2d"],
        ["tomato_soup_can_5.pkl", "05af029a062d57d21a008868b5eb0528"],
        ["tomato_soup_can_6.pkl", "bae913ca6997a290f81598fe3333b8bc"],
        ["tomato_soup_can_7.pkl", "d9ef195bce69b6570678a0d2c60dacd3"],
        ["tomato_soup_can_8.pkl", "ceea3478a8bbd45bfb418c9d06707a10"],
        ["tomato_soup_can_9.pkl", "921ee0fae5b6734c1925fae4229c963b"],
        ["tomato_soup_can_10.pkl", "c2cdd13c3d1e49cbd4ec9af1d0d492bd"],
        ["tomato_soup_can_11.pkl", "8f417acda5609380a92b90e27736d750"],
        ["tomato_soup_can_12.pkl", "e417d061afbb85c2381b62f695e35e9a"],
        ["tomato_soup_can_13.pkl", "65e6cdddf2ac6cfdb85d9a64e8c768ab"],
        ["tomato_soup_can_14.pkl", "7aeff557861205a2bc2c2d63190c956c"],
        ["tomato_soup_can_15.pkl", "69fe57827cdad6eb50b8fe801b65d02e"],
        ["tomato_soup_can_16.pkl", "07b33d03c556e7caa57bf8ebc9cb2fcd"],
        ["tuna_fish_can_1.pkl", "24f72eb374dbf648920467ce881c86ac"],
        ["tuna_fish_can_2.pkl", "d9190120881843abdd3b4127310c71d4"],
        ["tuna_fish_can_3.pkl", "89c625727ab4b86b418920cca8b91cc4"],
        ["tuna_fish_can_4.pkl", "bd5d07597f1d7758cf47d3a3f84af6b6"],
        ["tuna_fish_can_5.pkl", "6bfec091eaf4d1be6c13887ddd31332d"],
        ["tuna_fish_can_6.pkl", "f83e7cf9c380a9c1f9f9ef95bc3aef49"],
        ["tuna_fish_can_7.pkl", "267041cc7d58fbd1628d24490f711743"],
        ["tuna_fish_can_8.pkl", "c4645723df4e64ffc5b8f3af6f09cf90"],
        ["tuna_fish_can_9.pkl", "a61fca44064e7f4d8b20da6ca36865df"],
        ["tuna_fish_can_10.pkl", "48001d6d8a4da90c4f3ba46b7bf736f5"],
        ["tuna_fish_can_11.pkl", "56f345829b240af658245fcda6359a96"],
        ["tuna_fish_can_12.pkl", "9146a4b7bd085948919f4cfb3e8b3501"],
        ["tuna_fish_can_13.pkl", "ae0cf97255a31017c7655a36f35baa28"],
        ["tuna_fish_can_14.pkl", "f4a04961c85a67ffe1b1e7258a182c41"],
        ["tuna_fish_can_15.pkl", "edf98191ec87877dd983200cd2edd42c"],
        ["tuna_fish_can_16.pkl", "9cf60ce4c64c2e4ccc2e1259fc7cfa03"]
        ]


    def __init__(
        self, 
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        shuffle: bool = True,
        ) -> None:
        
        super().__init__(root, transform=transform, target_transform=target_transform)


        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.train = train

        self.data: Any = []
        self.targets = []

        meta_file_path = os.path.join(self.root, self.base_folder, "meta.pkl")
        with open(meta_file_path, 'rb') as fp:
            self.meta = pickle.load(fp, encoding="latin1")


        holdout_objects = self.meta["test_split"]
        self.holdout_objects = set(holdout_objects)

        self.object_instances = set([os.path.splitext(fl)[0] for fl in os.listdir(os.path.join(self.root, self.base_folder))
                                                            if fl != "meta.pkl"])

        if self.train:
            self.object_instances = self.object_instances.difference(self.holdout_objects)
        else:
            self.object_instances = self.object_instances.intersection(self.holdout_objects)

        for obj_inst in self.object_instances:
            obj_file_path = os.path.join(self.root, self.base_folder, obj_inst+".pkl")
            with open(obj_file_path, 'rb') as fp:
                entry = pickle.load(fp, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data)

        if shuffle:
            random_shuffle = random.Random(0)
            self.targets = np.array(self.targets)

            shuffle_indices = np.arange(self.targets.shape[0])
            random_shuffle.shuffle(shuffle_indices)

            self.targets = list(self.targets[shuffle_indices])
            self.data = self.data[shuffle_indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target_transform)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


    def _check_integrity(self) -> bool:
        for filename, md5 in self.object_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
