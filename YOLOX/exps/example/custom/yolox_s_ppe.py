#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # YOLOX-Small network scaling
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # SafeWatch AI classes (Helmet, No_helmet, Safety_shoes, No_SafetyShoes, Person)
        self.num_classes = 5

        # Point to the mapped COCO directories
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # RTX 3050 Hardware Optimizations
        self.data_num_workers = 4
        self.eval_interval = 5
        self.max_epoch = 50
