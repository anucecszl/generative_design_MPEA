# Generative Design of Multi-Principle Element Alloys (MPEA) using NSGAN

## Overview
This repository demonstrates the application of Non-Dominant Sorting Generative Adversarial Networks (NSGAN) in the generative design of multi-principle element alloys (MPEA). NSGAN, an innovative generative design framework, synergistically merges the data generation abilities of Generative Adversarial Networks (GAN) with the multi-objective optimization capabilities of the NSGA-II algorithm. Drawing from a comprehensive dataset of 1704 MPEA characteristics, this repository presents the complete workflow for applying the proposed framework.

## Repository Structure

### Dataset
- `dataset/dataset_format.py`: Formatting tools for the dataset.
- `dataset/dataset_parse.py`: Parsing utilities to process the raw dataset.
- `dataset/MPEA_mechanical_database.xlsx`: Comprehensive dataset containing 1704 MPEA characteristics.
- `dataset/empirical_parameter_calculator.py`: Tools for calculating empirical parameters.

### Generated Datasets
- `generated_dataset/`: Directory containing all generated databases.

### Property Prediction
- `property_prediction/`: Contains code implementations for predicting phase and mechanical properties using two random forest models.

### Saved Models
- `saved_models/`: Folder housing trained machine learning models.

### Generative Models
- `GAN_models/`: Code implementations of both GAN and NSGAN models.

### Online Application
- `streamlit/`: Implementation of the online application crafted using the Streamlit library.

## Features
- **Data Parsing & Engineering**: Comprehensive utilities for processing and formatting the MPEA dataset.
- **Property Prediction**: Random forest models to predict the phase and mechanical properties of the alloys.
- **Generative Adversarial Networks (GAN)**: Detailed implementation of the GAN for data generation.
- **NSGAN**: Core of the repository, illustrating the combined power of GAN and NSGA-II.
- **Online Application**: Interactive web application leveraging the NSGAN model, developed with Streamlit.

## Setup and Installation
1. Clone the repository:
git clone https://github.com/anucecszl/generative_design_MPEA.git
2. Navigate to the directory:
cd generative_design_MPEA
3. Install the required packages:
pip install -r requirements.txt

## Contributions

This project is a collaborative effort between Nick Birbilis and Zhipeng Li:

- **Zhipeng Li**: Responsible for the coding and development of the models.
  
- **Nick Birbilis**: Supervised the study and served as the lead Principal Investigator. 

## Contact

For any inquiries, please reach out via email at [u6766505@anu.edu.au](mailto:u6766505@anu.edu.au).
