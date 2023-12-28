
2. The images were resized to 360*360 for faster processing and
the results seemed to be similar to those with bigger image
size, the result images from the mode is 164*164.
3. For Data Augmentation all 5 of the techniques given in
Assignment Description were used:
1. Horizontal/Vertical Flip
2. Zooming/Cropping
3. Gamma Adjustment
4. Sheer Transformation
5.
Rotation
4.
Stayed true to the original architecture for UNet.
5.
Trained for 50 epochs,
Tried training from 30 epochs to 100 epochs, saw that by 30 epochs, model was still converging and with 100 epochs found that the optimal convergence was reached at around 50 epochs, hence the results were based on 50 epochs run.
6.
For batch size used 1 and for learning rate 1e-3 (0.001). Tried various batch sizes and found the 1 to be best for running times
and learning rate was
selected as 1e-3 after a lot of trial, also a small amount of momentum was also added to help the model overcome local minima.
7.
The training took about 5 minutes for running 50 epochs on Quadro 4000 series cluster.
Visual from Weights and biases to represent model details:
