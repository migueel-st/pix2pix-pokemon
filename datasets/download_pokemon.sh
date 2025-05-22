set -e
echo "Downloading Pokémon dataset from Kaggle..."
curl -L -o ./pokemon-images-and-types.zip\
  https://www.kaggle.com/api/v1/datasets/download/vishalsubbiah/pokemon-images-and-types
unzip -o ./pokemon-images-and-types.zip -d ./datasets/pokemon
rm ./pokemon-images-and-types.zip
echo "Pokémon dataset downloaded and extracted to ./datasets/pokemon ✅"
python datasets/prepare_pokemon_dataset.py