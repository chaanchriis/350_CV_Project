# IAT 360 Computer Vision Project

This computer vision project was done by @jjrbf and @chaanchriis for IAT 360 in the Fall 2024 semester. It is a trained model to detect trash using computer vision, aiming to detect trash in the context of parks.

## Credits

We used this [dataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset), along with images taken ourselves to train our model. See `iat360_cv_project.ipynb` for further credits for code creation.

## Description

- `/Final_data` contains all of the labels, but only the images that is not part of the Kaggle dataset. (001500.JPG - 001560.JPG)
- `/runs/detect` contains the different times we trained the model. The initial model can be found in `/runs/detect/train5/weights/best.pt` while the final model can be found in `/runs/detect/train6/weights/best.pt`.
- `computerVisionProject.py` is the python file used to train the model.
- `iat360_cv_project.ipynb` is the jupyter notebook used for preprocessing and checking for results. See this file for a more detailed overview of how we went through the project.
- `yolov8s.pt` is the pretrained model from YOLO.

## Project Description from Canvas:

> This assignment aims to apply the skills and knowledge you have acquired about image datasets, the YOLO object detection framework, and image annotation tools. You are required to either create a new image dataset or annotate an existing dataset, focusing on one of the following computer vision tasks: classification, detection, segmentation, or tracking.

## Tasks

**1. Identify a Problem:**

Choose a problem or topic in computer vision that interests you.
The problem should align with one of the following tasks:
Classification, Detection, Segmentation, Pose estimation, Tracking

**2. Dataset Creation/Enhancement:**

 Gather available datasets online and add 30 new images/labels per person. 2/3 of this data should be used as training data, the rest is for evaluation. The structure of the dataset and annotations should be correctly added in the sub-folder structure for YOLO. This new data, if it does not have any privacy concerns, should be added to github. If privacy & security cannot be guaranteed, share a link to a zip folder with the correct sub-folder structure. If you're using another dataset in addition to yours, make sure you have sharing rights. If you do not have sharing rights, do not share it in neither github or as .zip, but rather share the link of the dataset you found it from.

 - Step 1 - Find Existing Data:
Collect and assemble a dataset of images relevant to your chosen problem.
Ensure the dataset is diverse and representative of the problem.
 - Step 2 - Create New Data:
Create new images, 30 images per person, annotate the data using given annotation tools
 - BONUS STEP:
Perform data augmentation to increase your dataset size!

**3. Fine-Tuning YOLO with Your Custom Dataset:**

Clone the YOLO repository from its official GitHub page or the version specified in the course materials.
Integrate your custom dataset into the YOLO framework (transfer learning).
Fine-tune the YOLO model using your dataset. This involves adjusting the model's parameters to better suit your specific data.
Save your fine-tuned model, and evaluate the accuracy.

**4. Report:**

Provide a document outlining:
Your chosen problem and its significance.
Dataset report:
Explain how you created your data, this includes the steps to create the dataset above. #add links for annotation tools or any existing images.  
The methodology used for collecting and annotating the dataset.
Example snippets from your dataset (showing both images and labels)
Bias and privacy issues you see with the datasets you used.
Model analysis:
A detailed evaluation of your model’s performance.
A critical analysis of the model's shortcomings and ethical considerations.
Challenges faced during any of above steps and how they were addressed.

**5. Submission Format:**

Submit your report as a PDF file on canvas, the PDF should include the links to code & datasets in github. Make sure your links work properly.  