-> What We Did – Virtual Try-On Shirt App

1. Installed packages
    We installed OpenCV MediaPipe and NumPy using pip to handle webcam input pose and hand detection and image processing

2. Set up folders
    We created a resources shirts folder and added transparent PNG shirt images We also added a button png image to the main directory

3. Used webcam
    The app captures live video from the webcam to create an interactive experience

4. Detected pose and hand
    We used MediaPipes Pose and Hands modules to track shoulder hip and index finger landmarks in real time

5. Calculated shirt position
    Based on shoulder and hip positions we calculated where and how big the shirt overlay should be

6. Overlayed the shirt
    We blended the shirt image onto the users body using alpha transparency to make it look realistic

7. Added button control
    A virtual button appears at the bottom right corner When the index finger hovers over it the shirt changes

8. Displayed the UI
    We used MediaPipes drawing tools and OpenCV to show landmarks and overlays for a better user experience

-> How to Run the App

1. Clone the repo
    git clone https://github.com/gagandeep1763/Virtual-Try-on-Shirts

2. Install dependencies
    pip install opencv python mediapipe numpy

3. Add resources
    Place shirts in the resources shirts folder
    Place button png in the main directory

4. Run the project
    python main.py

5. Controls
    Show your index finger near the bottom right virtual button to switch shirts
    Press q on your keyboard to quit the app

