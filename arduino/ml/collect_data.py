#!/usr/bin/env python3
"""
collect_data.py — Collect real glove sensor data
BLUETOOTH / ARDUINO VERSION
"""

import serial
import csv
import time
import os
import sys
import numpy as np

# --------------------------------------------------
# Bluetooth / Serial settings
# --------------------------------------------------

BT_PORT = "COM5"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 3

# --------------------------------------------------
# File settings
# --------------------------------------------------

SAVE_PATH = "ml/data/gesture_data.csv"

FEATURE_COLS = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky"
]

ACTIVE_COLS = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky"
]

# Number of samples collected for each gesture
SAMPLES_PER_GESTURE = 60

# Arduino UNO ADC range
ARDUINO_ADC_MAX = 1023

# Convert Arduino values to 0–4095 range
NORMALIZED_FLAT = 0
NORMALIZED_BENT = 4095
NORMALIZED_RANGE = 4095

SCALE_FACTOR = NORMALIZED_RANGE / ARDUINO_ADC_MAX

# --------------------------------------------------
# Gestures
# --------------------------------------------------

GESTURES = [
    ("Hello", "All fingers fully FLAT/STRAIGHT — open palm"),
    ("Yes", "All fingers fully BENT — tight fist"),
    ("No", "Index+thumb flat, middle+ring fully bent"),
    ("ThankYou", "All fingers at HALFWAY position (medium bend)"),
    ("Please", "Thumb bent inward, other fingers only slightly bent"),
    ("Sorry", "Thumb+middle+ring fully bent, only index pointing up"),
    ("Help", "Thumb fully flat/straight, all other fingers fully bent"),
    ("Stop", "Thumb+ring bent, index+middle pointing up straight"),
    ("Good", "Index bent inward, middle+ring fully flat/straight"),
    ("Bad", "Index+middle bent, thumb+ring fully flat/straight"),
    ("Come", "Thumb+middle bent, index+ring straight"),
    ("A", "Maximum tight fist — all as bent as possible"),
    ("B", "Thumb bent only, all others fully straight"),
    ("C", "All fingers in medium C-curve — same bend on all"),
    ("D", "Index pointing up straight, middle+ring fully bent"),
    ("L", "Thumb+index straight (L shape), middle+ring fully bent"),
    ("V", "Thumb+ring bent, index+middle straight up (peace sign)"),
    ("S", "Thumb over fist — slightly less tight than A"),
    ("F", "Thumb+index pinched together, middle+ring straight up"),
]


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def parse_sensor_line(line):
    """Convert Arduino CSV line into five sensor values."""

    try:
        values = [float(x.strip()) for x in line.split(",")]

        if len(values) != 5:
            return None

        # Convert Arduino 0–1023 values to approximately 0–4095
        values = [int(round(v * SCALE_FACTOR)) for v in values]

        return values

    except (ValueError, TypeError):
        return None


def open_serial():
    """Open Bluetooth serial connection."""

    print(f"\nConnecting to Bluetooth device on {BT_PORT}...")

    try:
        ser = serial.Serial(
            BT_PORT,
            BAUD_RATE,
            timeout=SERIAL_TIMEOUT
        )

        time.sleep(2)
        ser.reset_input_buffer()

        print("Bluetooth connection successful.")
        return ser

    except serial.SerialException as e:
        print("\nERROR: Could not open serial port.")
        print(f"Port: {BT_PORT}")
        print(f"Details: {e}")
        print("\nCheck:")
        print("1. HC-05 is paired with the computer.")
        print("2. Correct COM port is being used.")
        print("3. Bluetooth device is connected.")
        print("4. Arduino is powered.")
        sys.exit(1)


def read_sensor_sample(ser):
    """Read one valid sensor sample."""

    start_time = time.time()

    while time.time() - start_time < SERIAL_TIMEOUT:

        try:
            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            values = parse_sensor_line(line)

            if values is not None:
                return values

        except Exception:
            continue

    return None


# --------------------------------------------------
# Main data collection
# --------------------------------------------------

def main():

    print("=" * 60)
    print("SIGN LANGUAGE GLOVE DATA COLLECTION")
    print("=" * 60)

    print(f"\nNumber of gestures: {len(GESTURES)}")
    print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
    print(f"Bluetooth port: {BT_PORT}")
    print(f"Baud rate: {BAUD_RATE}")

    # Create data directory
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Connect to Bluetooth
    ser = open_serial()

    all_data = []

    try:

        for gesture_index, (gesture, description) in enumerate(GESTURES, start=1):

            print("\n" + "=" * 60)
            print(
                f"Gesture {gesture_index}/{len(GESTURES)}: "
                f"{gesture}"
            )
            print("=" * 60)

            print(f"Position: {description}")

            input(
                "\nPlace your hand in the position and press ENTER "
                "to start collecting..."
            )

            samples = []

            print(
                f"\nCollecting {SAMPLES_PER_GESTURE} samples..."
            )

            while len(samples) < SAMPLES_PER_GESTURE:

                values = read_sensor_sample(ser)

                if values is None:
                    print(
                        "\nWarning: No valid sensor data received."
                    )
                    continue

                samples.append(values)

                print(
                    f"\rSamples collected: "
                    f"{len(samples)}/{SAMPLES_PER_GESTURE}",
                    end=""
                )

                time.sleep(0.02)

            print("\nCollection complete.")

            # Store samples
            for values in samples:

                row = values + [gesture]

                all_data.append(row)

            # Show average sensor values
            averages = np.mean(
                np.array(samples),
                axis=0
            )

            print("\nAverage sensor values:")

            for name, value in zip(
                FEATURE_COLS,
                averages
            ):
                print(
                    f"  {name:>6}: {value:.1f}"
                )

    except KeyboardInterrupt:

        print("\n\nData collection stopped by user.")

    finally:

        ser.close()

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    if len(all_data) == 0:

        print("\nNo data collected.")
        return

    print("\nSaving collected data...")

    with open(
        SAVE_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            FEATURE_COLS + ["label"]
        )

        writer.writerows(all_data)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)

    print(f"\nTotal samples collected: {len(all_data)}")
    print(f"Saved to: {SAVE_PATH}")

    print("\nNext step:")
    print("Train the machine-learning model using:")

    print("\npython ml/train_model.py --no-download")


if __name__ == "__main__":
    main()
