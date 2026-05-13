# Coding II Final Project
# Yellow ball (fuel) image identification

This project uses Pytorch machine learning to identify if a given image contains a yellow ball (fuel).

Data used: https://universe.roboflow.com/10014rebellion/rebuilt
 - Original data classifies where in the image the fuel is located
 - Data is processed by splitting each image into 4 quarters, allowing for images with and without fuel
 - Data is classified as fuel (F at the beginning of the filename in validation images) or no fuel
 - Some edge cases were deleted (if a fuel was only very slightly in the image)
 - Data is stored in a custom dataset
 - Model weights from 50 epochs are saved so model does not have to be retrained

 Target: identify which images have fuel
 Success: Over 94% success within validation images (3 incorrect out of 53), with the incorrect classifications having distinguishable features or being edge cases
 Improvements: Can be trained with more images and more image variation, as it seems to mistake large portions of white in an image as fuel

 To see image results: run network_test.py and it will loop through valid-quarters and display any incorrect predictions
