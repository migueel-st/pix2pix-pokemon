#!/bin/bash
set -e
echo "Downloading Pokémon dataset from Kaggle..."
curl -L -o ./pokemon-images-and-types.zip\
  https://www.kaggle.com/api/v1/datasets/download/vishalsubbiah/pokemon-images-and-types
mkdir prepare/
unzip -o ./pokemon-images-and-types.zip -d ./prepare/pokemon-images-and-types
rm ./pokemon-images-and-types.zip
echo "Pokémon dataset downloaded and extracted to ./prepare/pokemon-images-and-types ✅"