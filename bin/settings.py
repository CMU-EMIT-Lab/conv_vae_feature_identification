"""
INFO
File: settings.py
Created by: William Frieden Templeton
Date: January 27, 2023
"""


class TrainParams:
    def __init__(self,
                 epochs=100,
                 parent_dir='test',
                 name='my_model',
                 image_size=64,
                 batch_size=16,
                 latent_dim=128,
                 num_examples_to_generate=16,
                 learning_rate=0.0001,
                 section_divisibility=10,
                 test_train_split=4,
                 bright_sample=True,
                 dofolds=True,
                 kfolds=5
                 ):
        self.epochs = epochs
        self.parent_dir = parent_dir
        self.name = name
        self.image_size = image_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_examples_to_generate = num_examples_to_generate
        self.learning_rate = learning_rate
        self.section_divisibility = section_divisibility
        self.test_train_split = test_train_split
        self.bright_sample = bright_sample
        self.dofolds = dofolds
        self.kfolds = kfolds
