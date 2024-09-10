# Snail Tracker

This is a comprehensive project designed to track the behaviors of garden snails in their habitat using computer vision and machine learning models. The project aims to capture various environmental data points and behavioral patterns of snails, ultimately contributing to a better understanding of their daily activities.

## Current Developments

1. **Text Extraction from Digital Thermometer and Hygrometer:**
   - Developing computer vision models (like Tesseract OCR and potentially more robust deep learning models such as CRNN or Donut OCR) to extract `temperature` and `humidity` readings from the [Mini Digital Temperature Humidity Meter](https://www.amazon.com/dp/B07GNMKYCZ?ref=ppx_yo2ov_dt_b_fed_asin_title). 
   - Experimenting with various image pre-processing techniques, including contrast enhancement, noise reduction, and thresholding, to improve OCR accuracy.
   
2. **AWS Integration for Model Training and Deployment:**
   - Utilizing the AWS Toolkit for Visual Studio Code to seamlessly connect to AWS services such as S3 and SageMaker.
   - Creating an AWS S3 bucket (`snail-object-detection`) to store and manage the data (e.g., videos, images, datasets) collected from the snail habitat.
   - Configuring and deploying Amazon SageMaker for training the computer vision models on extracting environmental data (`temperature` and `humidity`) and performing object detection tasks.

3. **Data Pipeline Development:**
   - Setting up a data pipeline to automatically ingest footage from the [Tapo TP-Link 2K QHD Security Camera](https://www.amazon.com/dp/B0CH45HPZT?ref=ppx_yo2ov_dt_b_fed_asin_title) into AWS S3 every 15 seconds for analysis.
   - Developing scripts to process the video footage, extract relevant frames, and run them through machine learning models to gather insights about the snail's environment and behaviors.
   - Creating a dataset with columns such as `temperature`, `humidity`, `is-eating`, `is-drinking`, `movement`, and `preferred food types` using object detection and activity recognition models.

## Utilized Technology

- **Camera and Recording:**
  - [Tapo TP-Link 2K QHD Security Camera](https://www.amazon.com/dp/B0CH45HPZT?ref=ppx_yo2ov_dt_b_fed_asin_title): Captures high-resolution 2K footage of the snail habitat, stored locally on a micro-SD card and uploaded to AWS S3 for processing.

- **Environmental Sensors:**
  - [Mini Digital Temperature Humidity Meters](https://www.amazon.com/dp/B07GNMKYCZ?ref=ppx_yo2ov_dt_b_fed_asin_title): Records real-time temperature and humidity data within the snail enclosure, which is extracted using trained computer vision models.

- **Storage:**
  - [Amazon Basics Micro SDXC Memory Card](https://www.amazon.com/dp/B08TJZDJ4D?ref=ppx_yo2ov_dt_b_fed_asin_title): Used in conjunction with the Tapo Camera to locally store video footage.

## Snail Care

- **Enclosure:**
  - [REPTIZOO Small Glass Tank 8 Gallon](https://www.amazon.com/dp/B083PX9YR6?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1): A cubic foot glass terrarium providing a controlled environment for up to six snails, allowing them sufficient space to move and thrive.

- **Enrichment:**
  - [12 Inch Pet Snail Climbing Toys](https://www.amazon.com/dp/B0CWNYQ43M?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1): Provides climbing surfaces and stimulation for the snails, promoting natural behaviors such as exploration.

- **Humidity Control:**
  - [Continuous Empty Ultra Fine Plastic Water Mist Sprayer](https://www.amazon.com/dp/B0948WBX9L?ref=ppx_yo2ov_dt_b_fed_asin_title): Helps maintain a high-humidity environment, which is essential for snail activity and comfort.

- **Substrate:**
  - [Natural Coconut Fiber Substrate](https://www.amazon.com/dp/B0BWRHB88C?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1): Provides a burrowing medium for snails and aids in maintaining appropriate moisture levels in the tank.

- **Water Supply:**
  - [Water Bowl](https://www.amazon.com/dp/B08GNZ4737?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1): Offers a safe and accessible water source for snails to drink and bathe, minimizing the risk of shell damage.

- **Food:**
  - [Pet Land Snail Food](https://www.amazon.com/dp/B0B8QC4B8X?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1): A calcium-rich diet to support shell growth and overall health.

## Future Directions

- **Refining Data Collection Intervals:**
  - Optimize the data collection interval (e.g., every 15 seconds) to balance data richness and storage efficiency.

- **Advanced Behavioral Analysis:**
  - Train object detection models to recognize specific snail behaviors (e.g., `is-eating`, `is-drinking`, `is-climbing`) and track movement patterns.
  - Develop algorithms to analyze food preferences based on observed interactions.

- **Improving Model Accuracy:**
  - Experiment with different computer vision models and architectures (e.g., YOLO, SSD, EfficientDet) to improve the accuracy of environmental data extraction and behavioral classification.

- **Deployment on Edge Devices:**
  - Investigate deploying trained models on edge devices to reduce latency and reliance on cloud infrastructure.

## Repository Structure

```plaintext
├── env/
├── models/
├── outputs/
├── processed-videos/
├── raw-videos/
├── training-images/
├── .gitignore
├── image.ipynb
├── IMG_3741.mov
├── IMG_3741.mp4
├── ocr-donut.ipynb
├── openai.ipynb
├── README.md
├── requirements.txt
├── snail_data.csv
├── temp_roi.png
└── tutorial.ipynb