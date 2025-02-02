import cv2
import numpy as np

class OpticalFlow:
    def __init__(self, video_path, output_path="output.avi", window_size=5, bg_frames=50, fps=30):
        # class attributes
        self.video_path = video_path  #input video
        self.output_path = output_path  #output video
        self.window_size = window_size  # window size for lucas kannade algorithm
        self.bg_frames = bg_frames    #no. of frames for initializing background
        self.fps = fps      # fps for output video

        self.cap = cv2.VideoCapture(video_path)    # opening the input video
        if not self.cap.isOpened():
            raise ValueError("Can't open the video")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #video width
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   #video height
        self.frame_size = (self.frame_width, self.frame_height)      #video size

        #video writer to store output video of the same size as input.
        self.out = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.frame_width, self.frame_height)
        )

        self.prev_frame = None  # initializing prev frame
        self.background = None  # initializing background

    def initialize_background(self):  #set background of the video as the median of the first bg_frames frames
        frames = []
        for _ in range(self.bg_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame) # store the first bg_frames number of frames in  grayscale format in frames list

        self.background = np.median(frames, axis=0).astype(np.uint8)   # set the background as the median of the frames stored above
        print("Background initialized.")

    def grad_matrix(self, prev_frame, current):

        # compute Ix, Iy and It
        Ix, Iy = np.zeros_like(prev_frame, dtype=np.float32), np.zeros_like(prev_frame, dtype=np.float32)

        It = current - prev_frame # difference between 2 consecutive frames
        for i in range(1, prev_frame.shape[0] - 1):   # loop through width
            for j in range(1, prev_frame.shape[1] - 1):   # loop through height
                Ix[i, j] = (prev_frame[i, j + 1] - prev_frame[i, j - 1]) / 2   # finding gradient in x dir
                Iy[i, j] = (prev_frame[i + 1, j] - prev_frame[i - 1, j]) / 2   # finding gradient in y dir
        return Ix, Iy, It

    def lucas_kanade(self, prev_frame, current, window_size):

        Ix, Iy, It = self.grad_matrix(prev_frame, current)  # gradient matrix of the prev frame
        
        h_vel, v_vel = np.zeros_like(prev_frame), np.zeros_like(prev_frame)  # initialize the velocities of each pixel values

        wsizeby2 = window_size // 2

        # sliding window to find the A and B matrix using the gradient matrix of the image

        for i in range(wsizeby2, prev_frame.shape[0] - wsizeby2):
            for j in range(wsizeby2, prev_frame.shape[1] - wsizeby2):
                Ix_matrix = Ix[i - wsizeby2:i + wsizeby2 + 1, j - wsizeby2:j + wsizeby2 + 1].flatten()
                Iy_matrix = Iy[i - wsizeby2:i + wsizeby2 + 1, j - wsizeby2:j + wsizeby2 + 1].flatten()
                B = It[i - wsizeby2:i + wsizeby2 + 1, j - wsizeby2:j + wsizeby2 + 1].flatten() # B matrix

                A = np.stack((Ix_matrix, Iy_matrix), axis=1)  # A matrix
                A_tr = A.T

                AtA = np.dot(A_tr, A)
                AtB = np.dot(A_tr, B)

            # calculating u matrix
                if np.linalg.det(AtA) != 0:
                    u = np.dot(np.linalg.inv(AtA), AtB)
                    h_vel[i, j] = u[0]
                    v_vel[i, j] = u[1]
                else:
                    h_vel[i, j] = 0  
                    v_vel[i, j] = 0
        return h_vel, v_vel  # return the velocities

    def process_video(self):
        self.initialize_background()   # call initialize background function

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # convert current frame to grayscale
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #  subtract the background frame from the current frame
            fg_mask = cv2.absdiff(self.background, current)
            _, fg_mask = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)

            #  overlay
            overlay = frame.copy()
            overlay[fg_mask == 255] = (0, 255, 0)  # Light green where mask is white

            # overlay with the original frame
            combined_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            # optical flow computation 
            if self.prev_frame is not None:
                h_vel, v_vel = self.lucas_kanade(self.prev_frame, current, self.window_size)

            # output video
            self.out.write(combined_frame)

            cv2.imshow("Foreground Overlay", combined_frame)

            self.prev_frame = current

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.endit()

    def endit(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.avi"  
    output_path = "output.avi"
    optical_flow = OpticalFlow(video_path, output_path)
    optical_flow.process_video()