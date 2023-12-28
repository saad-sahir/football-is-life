# football-is-life

### weights lcnn: https://drive.google.com/file/d/1NvZkEqWNUBAfuhFPNGiCItjy4iU0UOy2
### weights lcnn path: './lcnn/weights.pth.tar'

----------------------------------------------

football-is-life is a computer vision annotation tool for football analytics that idenitifies and annotates the boundaries of a football pitch and tracks the position of each player in that pitch and maps those two to a 2D render of the football pitch

## Flow of Processes

### The input

The input for the model will be a frame from the video stream of a football match. For our research images, we extracted a frame from the 2018 World Cup match between Spain and Portugal.

![input frame](https://github.com/saad-sahir/football-is-life/blob/main/images/test.jpg)

## The Line Segment Detector

Our first approach used native cv2 methods to extract as much line information from the image as possible. We did this by first masking the pitch from the stands and the ads, then the image with a gray mask gets fed into a Canny Edge Detector, the output of which gets fed into a Hough Lines Transform.

![masked pitch](https://github.com/saad-sahir/football-is-life/blob/main/results/mask_test.png)

![lines detected from image](https://github.com/saad-sahir/football-is-life/blob/main/results/limage.png)

The problem with this approach for now is that the Hough Lines Transform doesn't provide us with singular lines for the pitch boundaries but rather smaller lines that when put together, give us the full line. We attempted to cluster the lines by slope and intercept but that yielded more confusing results that we couldn't proceed with.

## LCNN Approach

We encountered a [GitHub repo](https://github.com/zhou13/lcnn) for a custom trained CNN for end-to-end wireframing that yielded very promising results in what it detected on the image but it took minutes (on the free GPU on colab) to just render 1 single frame. 

![lines detected from LCNN](https://github.com/saad-sahir/football-is-life/blob/main/results/lcnn_test1.png)

As you can see, it detects the gorund truth lines essentially perfectly but with the cost of noise. So we added the green mask code from the LSD approach and ran it again.

![lines detected from LCNN with mask](https://github.com/saad-sahir/football-is-life/blob/main/results/lcnn_test1_masked.png)

This yielded good results but the computational time was too long
