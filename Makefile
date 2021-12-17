.PHONY: prepare-pip prepare-pacman-nvidia destroy clear all

all: build/images
	python main.py

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

build/images: build
	mkdir -p build/images
