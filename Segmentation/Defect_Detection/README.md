# Defect Detection in Steel using Unet CNN Articture

-This is a very simple example of defect segmentation in steel <br>
-I am using "resnet18" as a backbone which act as an encoder in Unet architecture. the encoder weights were trained using "imagenet".
-The dataset used is "Severstal: Steel Defect Detection" which have 4 classes for for simplicity i just used all the defect as a single class defect.

-The model was trained for 3 epochs using "BCEWithLogitsLoss" function and achieved around 95% accuracy on validation dataset.



![Screenshot from 2023-10-22 17-55-43](https://github.com/abdulnim/Pytorch-Projects/assets/113373212/3b85c592-1110-4be4-b9b3-d98a44f5fea5)




