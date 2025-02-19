# Real-time Face Recognition using dlib

This project implements a real-time face recognition system using the dlib library, OpenCV, and Python. The system can detect and recognize faces in real-time using your PC's webcam by comparing them against a pre-built database of known faces.

## Features

- Real-time face detection and recognition using webcam
- Face database management and storage
- CSV-based face descriptor storage
- High accuracy using dlib's face recognition model
- Support for multiple face recognition

## Project Structure

- `real_time_v2.py`: Main script for real-time face recognition using webcam feed
- `my_dlib_funcs.py`: Core functions for face detection, recognition, and database operations
- `image_recognition.py`: Script for processing and recognizing faces in static images
- `generate_db_csv.py`: Utility script to generate face database from input images
- `people.csv`: Database file containing face descriptors and identities
- `requirements.txt`: Project dependencies

### Directories
- `database/`: Contains face images used to build the recognition database
- `models/`: Contains dlib's pre-trained models
- `Outputs/`: Stores output files and results
- `testing imgs/`: Test images for validation

## Requirements

- Python 3.10.8
- dlib 19.24.6
- numpy 2.2.3
- opencv-python 4.11.0.86

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. First, populate the `database` directory with face images you want to recognize
2. Generate the face database:
   ```
   python generate_db_csv.py
   ```
3. Run the real-time recognition:
   ```
   python real_time_v2.py
   ```
4. Run the static image recognition:
   ```
   python image_recognition.py "path/to/image.jpg"
   ```

## License

This project is licensed under the terms included in the LICENSE file.