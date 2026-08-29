"""Register the fixed 16-task RoboCasa365 comparison panel."""

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY


TASK_SET_REGISTRY["amd_match16"] = [
    "ArrangeTea",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "DeliverStraw",
    "OpenCabinet",
    "OpenDrawer",
    "PackIdenticalLunches",
    "PickPlaceDrawerToCounter",
    "PortionHotDogs",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SeparateFreezerRack",
    "SlideDishwasherRack",
]
