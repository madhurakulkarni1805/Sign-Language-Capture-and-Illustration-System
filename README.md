# Sign Language Capture and Illustration System with Sensor Integration

##  Overview

The **Sign Language Capture and Illustration System with Sensor Integration** is a smart-glove-based system designed to recognize hand gestures using flex sensors and machine learning.

The system uses **five flex sensors** to capture finger movements. An **Arduino UNO** reads the sensor values and transmits the data wirelessly through an **HC-05 Bluetooth module**. A **Raspberry Pi** is used for data processing and machine learning. The collected sensor data is used to create a dataset and train a machine learning model for gesture recognition.

---

##  Objectives

- Capture finger movements using flex sensors.
- Transmit sensor data wirelessly using Bluetooth.
- Collect and prepare a gesture dataset.
- Train a machine learning model using the collected dataset.
- Recognize different hand gestures.
- Provide a base for converting recognized gestures into meaningful output.

---

##  Hardware Requirements

- Arduino UNO
- Raspberry Pi
- 5 × Flex Sensors
- HC-05 Bluetooth Module
- Resistors
- Breadboard
- Jumper Wires
- Glove

---

##  Software and Technologies

- Arduino IDE
- Python
- NumPy
- Pandas
- Scikit-learn
- Machine Learning
- Bluetooth Communication
- CSV Data Processing
- Raspberry Pi

---

## 🔌 Hardware Connections

| Component | Arduino Pin |
|-----------|-------------|
| Thumb Flex Sensor | A0 |
| Index Flex Sensor | A1 |
| Middle Flex Sensor | A2 |
| Ring Flex Sensor | A3 |
| Pinky Flex Sensor | A4 |
| HC-05 TX | Digital Pin 2 |
| HC-05 RX | Digital Pin 3 |

---

## ⚙️ Working Principle

The system works through the following steps:

1. Five flex sensors are attached to the fingers of a glove.
2. The flex sensors detect the bending of each finger.
3. Arduino UNO reads the analog values from the sensors.
4. The sensor values are transmitted through the HC-05 Bluetooth module.
5. Raspberry Pi receives the sensor data.
6. Python is used to collect and store the sensor readings.
7. The collected readings are organized into a dataset.
8. The dataset is processed and used to train a machine learning model.
9. The trained model learns different patterns corresponding to hand gestures.
10. New sensor readings can be given to the trained model for gesture recognition.

---

##  Machine Learning

The system uses the readings from the five flex sensors as input features.

### Machine Learning Process

```text
Sensor Data
     ↓
Data Collection
     ↓
Dataset Creation
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Gesture Recognition
