# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

 - **Author:** YUVANESH HARI B Group 2
 - **Context:** Infosys Internship Project, August 2025  

---

## 📌 Project Overview  

**SafetyEye** is an AI system designed for **real-time workplace safety monitoring**. It leverages **computer vision** and a **YOLOv8 object detection model** to analyze video feeds and ensure workers are wearing required **Personal Protective Equipment (PPE)** — specifically **hardhats** and **safety vests**.  

The goal is simple but critical:  
- Automate safety compliance checks.  
- Reduce workplace accidents.  
- Provide a scalable, high-performance monitoring solution.  

The project also includes a **full data processing pipeline** to prepare a robust, clean dataset for training the detection model.  

---

## 🚀 Key Features  

- **Real-Time Detection:** Identifies `hardhat`, `vest`, and `worker` classes instantly from video feeds.  
- **Compliance Logic:** Built-in rule engine checks if each detected worker has the required PPE (hardhat + vest).  
- **Advanced Data Pipeline:**  
  - Data cleaning  
  - Visual deduplication  
  - Augmentation (flip, rotate, brightness, etc.)  
  - Train/validation/test split  
- **GPU Accelerated:** Harnesses CUDA-enabled GPUs for fast **training** and **inference**.
