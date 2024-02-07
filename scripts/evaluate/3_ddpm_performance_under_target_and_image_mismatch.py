"""
Assess how does denoising performance degrade when there is a mismatch between segmentation mask and the image 

We do this in three ways:
  1. for a given image, use another image from the same patient but from very different cut
  2. for a given image, find the segmentation masks from another patient around -5, +5 cuts. Most images are registered to the same space, so the shift will be due to patient's anatomy
  3. for a given image, find the segmentation masks for the same patient around -5, +5 cuts  

We set the baseline or the ceiling at the same performance as the original image and the original segmentation mask.

"""

# Seleccionar imagenes al asar del conjunto de train:

#    Encontrar imagenes que sean muy diferentes: