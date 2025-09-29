# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025
An AI-based system that uses video surveillance feeds to monitor workplace occupancy levels and detect Personal Protective Equipment (PPE) compliance, ensuring a safer and more efficient environment.

**🚀 Project Overview**
This project addresses two critical challenges in modern workplace management: ensuring employee safety and optimizing space utilization. By leveraging the YOLOv8 object detection model, the system analyzes real-time video feeds from construction sites or industrial spaces to automatically detect safety compliance violations, such as missing helmets or safety vests. The platform is designed to provide administrators with a real-time dashboard, offering actionable insights through visualizations and instant alerts to improve safety protocols and manage space effectively.

**Key Features**:-

Automated PPE Detection: Identifies workers and detects whether they are wearing essential safety gear like helmets, masks, and vests.

Real-time Violation Alerts: Generates instant notifications when a safety compliance violation is detected.

Occupancy Monitoring: Tracks the number of people in designated zones to help managers optimize space.

Data-driven Dashboard: A live dashboard for administrators to view compliance statistics, recent events, and overall safety trends.

**Dataset**
This project utilizes the Construction Site Safety Image Dataset available on Kaggle. It contains images with YOLO-formatted annotations for various classes related to personal protective equipment and construction site environments.

Source: Construction Site Safety Image Dataset on Kaggle

**🛠️ Getting Started**
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8 or higher

Pip (Python package installer)

1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

2. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage project dependencies.


3. Install Dependencies
Install all the required packages using the requirements.txt file.
pip install -r requirements.txt

4. Download the Dataset
Download the dataset from the Kaggle link.
Unzip the file and place its contents into a dataset/raw directory in the project root. 

5. Run the Preprocessing Script
The script will clean, split, and augment the dataset, preparing it for training.
After running, a new dataset/processed folder will be created with the prepared data.

🧠 Model Training
Once the dataset is preprocessed, you can begin training the YOLOv8 model.

The training results and the best model weights (best.pt) will be saved in a runs/ directory.

**💻 Technologies Used**
Python: Core programming language

OpenCV: For image processing

Ultralytics YOLOv8: The object detection model

Albumentations: For advanced image augmentation

NumPy: For numerical operations

Pillow: For image handling and verification

scikit-learn: For splitting the dataset