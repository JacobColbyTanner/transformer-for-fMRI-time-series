import cv2
import numpy as np


movies_path1 = "/N/project/networkRNNs/HCP_movie_stimulus/7T_MOVIE1_CC1_v2.mp4"
movies_path2 = "/N/project/networkRNNs/HCP_movie_stimulus/7T_MOVIE2_HO1_v2.mp4"
movies_path3 = "/N/project/networkRNNs/HCP_movie_stimulus/7T_MOVIE3_CC2_v2.mp4"
movies_path4 = "/N/project/networkRNNs/HCP_movie_stimulus/7T_MOVIE4_HO2_v2.mp4"





# Open the video file
cap = cv2.VideoCapture(movies_path4)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame Rate:", fps)

movie_gray = []
# Loop through each frame
time = 0
frame_counter = 0

while cap.isOpened():
    
    ret, frame = cap.read()

    
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_counter +=1

    if frame_counter == fps:
            
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (gray_frame.shape[1] // 20, gray_frame.shape[0] // 20))

        # Access pixel values (example: print the top-left pixel)
        #print(gray_frame[0, 0])
        

        movie_gray.append(gray_frame)

        time += 1
        print(time)
        frame_counter = 0

    # Optionally display the frame (for debugging purposes)
    #cv2.imshow('Frame', gray_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all frames
cap.release()
cv2.destroyAllWindows()

movie_gray = np.array(movie_gray)


np.save("movies/movie_gray_4.npy",movie_gray)

import matplotlib.pyplot as plt

print(movie_gray.shape)
plt.imshow(movie_gray[900,:,:])
plt.savefig("figures/movie_gray_sample4.png", dpi=300, format='png', bbox_inches='tight')
