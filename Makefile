.PHONY: prepare-pip prepare-pacman-nvidia destroy clear train test

prepare-pip:
	pip install numpy pandas matplotlib seaborn sklearn tensorflow tensorflow_hub

prepare-pacman-nvidia:
	sudo pacman -S --needed python cuda cudnn

env:
	python -m venv env

build:
	mkdir -p build

clear:
	rm -rf build

destroy: clear
	rm -rf env

build/model: build scripts/create_model.py scripts/constants.py
	mkdir -p build/model
	python scripts/create_model.py

train: build/model src/flowers
	python scripts/train_model.py

build/images: build
	mkdir -p build/images

build/images/collisions.png: build/images scripts/test_model.py
	python scripts/test_model.py

test: build/images/collisions.png
