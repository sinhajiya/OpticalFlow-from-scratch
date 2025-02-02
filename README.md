# OpticalFlow-from-scratch
This project implements the **Lucas-Kanade method** from scratch to compute **optical flow** in a video.  

The `main.py` file contains the code to find Optical flow in a video using the Lucas-Kanade method.
The result is stored in the ***output.avi* shows a video with green overlay on the region where motion *may* exist. 

## Steps  
1. Compute the **initial background** as the median of the first `n` frames (n, a user defined parameter).  
2. Compute the spatial and temporal image derivatives (**Iₓ, Iᵧ, Iₜ**).  
3. Apply the **Lucas-Kanade algorithm** to compute the motion vectors (**A, B, and velocity of flow**).  
4. Overlay the computed optical flow on the video and visualize the results.  

## Note  
- This method is computationally intensive and may be **slow** for large videos files such as the one given here. 
