import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

url = 'https://raw.githubusercontent.com/sussykeem/eos_predictor/refs/heads/main/eos_dataset/train_data.csv'

class MoleculeVisualizer():

    def visualize_molecule_2D(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) # add implicit hydrogens
        if mol is None:
            print("Invalid SMILES string.")
            return

        img = Draw.MolToImage(mol, size=(224, 224))

        return img # PIL image

path = 'smiles.csv'

data = pd.read_csv(url)

imgs = []

for smile in data['smile']:
    img = MoleculeVisualizer().visualize_molecule_2D(smile)
    print(type(img))
    imgs.append(img)


    # input = image
    # target = image

    # Encoder training: CNN
    # goal: takes an image and generates a molecular embedding
    # training: fit embedding such that embedding can reconstruct original image

    # Ezn 

    # image -> encoder -> encoding -> decoder -> image


    # Decoder training:

    # image -> frozen encoder -> encoder -> decoder -> 
