# Title_generation
This project is a simple CNN used to generate the string images. Here are some examples:

![image](https://github.com/Touyuki/Title_generation/blob/main/images/ABDE0EB.png)

![image](https://github.com/Touyuki/Title_generation/blob/main/images/TAXSSEQ.png)

![image](https://github.com/Touyuki/Title_generation/blob/main/images/WORLD00.png)


Step1: Split the images of 26 single letters and create the image of the space

python split.py

Step2: Generate the training data, the images of random strings

python random_make.py

Step3: Train the network

python generate.py

Step4: Test

python test.py
