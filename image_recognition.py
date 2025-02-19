"""
This script performs facial recognition on input images using the dlib library.
It reads a database of faces stored in a CSV file and compares input image faces against the database.
The script will display the recognized faces with bounding boxes and names of matched individuals.
"""
import time
import cv2
import dlib
import os
import argparse
from my_dlib_funcs import *

def process_image(image_path, predictor, face_rec, HOG_face_detector, db_face_descriptors, max_dist_thresh=0.7):
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Get face descriptors from the image
    descriptors = get_face_descriptors(frame,
                                     detection_scheme='HOG',
                                     shape_predictor=predictor,
                                     face_recognizer=face_rec,
                                     upsampling=1)
    
    # Recognize faces by comparing with database
    recognize(target_descriptors=descriptors,
             database_descriptors=db_face_descriptors,
             max_dist_thresh=max_dist_thresh)
    
    # Draw bounding boxes and labels for each detected face
    for desc in descriptors:
        # Get bounding box coordinates
        left = desc["bounding box"][0]
        top = desc["bounding box"][1]
        right = desc["bounding box"][2]
        bottom = desc["bounding box"][3]
        
        # Draw bounding box and name
        frame = cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=4)
        frame = cv2.putText(frame, desc["name"], (left - 10, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    return frame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial recognition on images')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, help='Path to save the output image (optional)')
    args = parser.parse_args()

    # Set working paths
    current_dir = os.getcwd()
    database_path = os.path.join(current_dir, 'database')
    output_path = os.path.join(current_dir, 'Outputs')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Set models paths
    shape_predictor_path = os.path.join(current_dir, 'models', 'shape_predictor_68_face_landmarks_GTX.dat')
    face_recognition_model_path = os.path.join(current_dir, 'models', 'dlib_face_recognition_resnet_model_v1.dat')
    cnn_model_path = os.path.join(current_dir, 'models', 'mmod_human_face_detector.dat')

    # Load the models
    predictor = dlib.shape_predictor(shape_predictor_path)
    face_rec = dlib.face_recognition_model_v1(face_recognition_model_path)
    HOG_face_detector = dlib.get_frontal_face_detector()

    # Read the database
    print("Loading face database...")
    db_face_descriptors = read_db_csv(filename='people.csv')
    print(f"Loaded {len(db_face_descriptors)} faces from database")

    try:
        # Process the image
        print(f"Processing image: {args.image_path}")
        result_image = process_image(args.image_path, predictor, face_rec, 
                                   HOG_face_detector, db_face_descriptors)

        # Display the result
        cv2.imshow("Result", result_image)
        
        # Save the result if output path is specified
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(output_path, 'result_' + os.path.basename(args.image_path))
        
        cv2.imwrite(output_file, result_image)
        print(f"Result saved to: {output_file}")
        
        # Wait for key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
