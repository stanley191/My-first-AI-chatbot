# My-first-AI-chatbot

A simple Python-based AI chatbot that understands user intents and responds appropriately using a neural network.

---

##  Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Training Your Own Model](#training-your-own-model)
- [Technologies & Dependencies](#technologies--dependencies)
- [Contribution](#contribution)
- [License](#license)

---

## Overview

Following the tutorial ["Build Your First Python Chatbot Project"] on DZone, this project implements an intent-based chatbot from scratch using machine learning. It processes user input, predicts the intent, and returns an appropriate response :contentReference[oaicite:1]{index=1}.

---

## Project Structure
├── intents.json # Intents with tags, patterns, and responses
├── train_chatbot.py # Script to train the model
├── gui_chatbot.py # GUI script to chat with the trained bot
├── chatbot_model.h5 # Saved Keras model
├── words.pkl # Vocabulary list (lemmatized words)
├── classes.pkl # List of intent tags
└── README.md # This file

---

## Features

- Tokenizes and lemmatizes patterns from `intents.json`
- Builds a neural network with 3 dense layers (128, 64 neurons, softmax output)
- Trains using SGD optimizer for intent classification
- Stores model in `chatbot_model.h5`; vocabulary and classes in pickled files
- Provides a simple GUI (via Tkinter) to interact with the chatbot :contentReference[oaicite:2]{index=2}
