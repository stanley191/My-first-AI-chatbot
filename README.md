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

---

## Overview

Following the tutorial "Build Your First Python Chatbot Project" on DZone, this project implements an intent-based chatbot from scratch using machine learning. It processes user input, predicts the intent, and returns an appropriate response.

---

## Project Structure
1 intents.json # Intents with tags, patterns, and responses
2 train_chatbot.py # Script to train the model
3 gui_chatbot.py # GUI script to chat with the trained bot
4 chatbot_model.h5 # Saved Keras model
5 words.pkl # Vocabulary list (lemmatized words)
6 classes.pkl # List of intent tags
7 README.md # This file

---

## Features

- Tokenizes and lemmatizes patterns from `intents.json`
- Builds a neural network with 3 dense layers (128, 64 neurons, softmax output)
- Trains using SGD optimizer for intent classification
- Stores model in `chatbot_model.h5`; vocabulary and classes in pickled files
- Provides a simple GUI (via Tkinter) to interact with the chatbot :contentReference[oaicite:2]{index=2}

## Setup & Installation

1. **Clone the repository**
   ```bash
    git clone https://github.com/stanley191/My-first-AI-chatbot.git
    cd My-first-AI-chatbot
2. **Create a virtual environment (recommended)**
   ```bash
    python3 -m venv venv
    source venv/bin/activate     # macOS/Linux
    venv\Scripts\activate        # Windows
3. Install dependencies
   ```bash
   pip install nltk keras tensorflow # also need pickles and numoy but are built in libraries to python
   
## Usage
1. Train the Model by running "Train_chatbot.py"
2. Run the Chatbot GUI

## Training Your Own Model

1. Customize or extend intents.json with new tags, patterns, and responses.
2. Re-run train_chatbot.py to retrain and update the model files.
3. Test it via the GUI (gui_chatbot.py).

## Technologies & Dependencies
1. Python 3.x
2. NLTK – Tokenization and lemmatization
3. Keras (with TensorFlow backend) – Building and training neural network
4. Tkinter – Chat GUI interface
5. NumPy, pickle – Data handling and serialization

## Contribution
Contributions are welcome!
1. Fork the project
2. Create a feature branch (git checkout -b feature/YourFeature)
3. Commit your improvements (git commit -m "Add your feature")
4. Push and open a pull request












   
