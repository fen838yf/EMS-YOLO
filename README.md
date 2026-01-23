EMS-YOLO Model Configuration Guide:
I. Data Preparation 
Before starting training, you need to prepare a labeled dataset. You can choose a public dataset for this purpose. Place the dataset in the EMS-YOLO\main\datasets folder of the project and modify the EMS-YOLO\main\data\ems.yolo.data.yaml configuration file to point to this dataset.
II. Model Environment Configuration
After data preparation, you need to configure the model's runtime environment. First, install the required software packages and dependencies for EMS-YOLO. You can use pip or conda for installation. Refer to the EMS-YOLO\main\requirements.txt file for the specific versions of the required packages and dependencies. Next, you can view the model architecture in the EMS-YOLO\main\cfg\training\ems-yolo.yaml file. Finally, since the EMS-YOLO model is based on the Mamba deep learning framework, training must be conducted on a Linux system or WSL (Windows Subsystem for Linux).
Following these steps, you should be able to use EMS-YOLO to train your own dataset and achieve satisfactory performance. Please note that this is a basic guide, and the specific implementation may vary depending on the environment and requirements.
The EMS-YOLO model code is associated with a paper published in the Journal of Supercomputing. You may optimize the model based on this work, but proper attribution to the source is required. 
