from .instance.aircraft import *
from .instance.car import *
from .instance.cub import *
from .instance.dog import *
from .instance.flower import *
from .instance.food import *
from .instance.pascal import *
from .instance.pet import *
from .instance.waterbird import *
from .instance.bear import *
from .instance.turtle import *
from .instance.python import *

DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugDataset,
    "flower": Flowers102Dataset,
    "car": CarHugDataset,
    "pet": PetHugDataset,
    "aircraft": AircraftHugDataset,
    "food": FoodHugDataset,
    "pascal": PascalDataset,
    "dog": StanfordDogDataset,
    "bear": BearHugDataset,
    "turtle": TurtleHugDataset,
    "python": PythonHugDataset,
}
IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDataset,
    "flower": FlowersImbalanceDataset,
}
T2I_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugDatasetForT2I,
    "flower": FlowersDatasetForT2I,
    "bear": BearHugDatasetForT2I,
    "turtle": TurtleHugDatasetForT2I,
    "python": PythonHugDatasetForT2I
}
T2I_IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDatasetForT2I,
    "flower": FlowersImbalanceDatasetForT2I,
}
