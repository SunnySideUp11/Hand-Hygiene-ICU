import cv2
import mediapipe as mp
import os
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_csv_results(IMAGE_FOLDER, OUTPUT_FOLDER, action_idx):
    print(IMAGE_FOLDER)
    OUTPUT = [] # Used to save the hand key information of all pictures in the entire current folder
    CSV_FILE = os.path.join(OUTPUT_FOLDER, f"{action_idx}.csv")
    IMAGE_FILES = os.listdir(IMAGE_FOLDER)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            image_file = os.path.join(IMAGE_FOLDER, file)
            image = cv2.flip(cv2.imread(image_file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
            if not results.multi_handedness: # If the hand cannot be detected, skip it
                continue
            
            Left, Right = False, False  # Used to distinguish between left and right hands

            num_hands = len(results.multi_handedness)
            if num_hands == 2:
                Left, Right = True, True
            elif num_hands == 1:
                if results.multi_handedness[0].classification[0].label == "Left":
                    Left = True
                else:
                    Right = True
                    
            output = {}
            output['Left'] = []
            output['Right'] = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):     
                if Left and idx == 0:
                    for data in hand_landmarks.landmark:
                        output["Left"] += [data.x, data.y, data.z]
                else:
                    for data in hand_landmarks.landmark:
                        output["Right"] += [data.x, data.y, data.z]
            OUTPUT.append(output) 
    df = pd.DataFrame(OUTPUT)
    df.to_csv(CSV_FILE)
    
    
if __name__ == "__main__":
    folders = [f"../data/image/view3/p{i}" for i in range(1, 9)]
    for p_idx, folder in enumerate(folders):
        output_folder = f"../data/landmark/view3/{p_idx + 1}"
        os.mkdir(output_folder)
        image_folders = [os.path.join(folder, str(i)) for i in range(1, 8)]
        for action_idx, image_folder in enumerate(image_folders):
            get_csv_results(image_folder, output_folder, action_idx + 1)