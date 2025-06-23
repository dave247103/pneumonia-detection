# An Intelligent System for Pneumonia Detection from Pediatric Chest X-Rays

**Course:** [595II] - Intelligent Systems  
**Author:** Dawid Stecko  
**University:** University of Pisa, MSc in Computer Engineering  
**Academic Year:** 2024/2025

---

## 1. Project Overview

This project presents an Intelligent System (IS) for the binary classification of pediatric chest X-ray images into **Normal** or **Pneumonia** categories. The system is designed not only to achieve high diagnostic accuracy but also to provide explainability for its predictions, a critical requirement for AI in medical applications.

The core of the system is built using deep learning models implemented in PyTorch. We compare the performance of two state-of-the-art Convolutional Neural Network (CNN) architectures, **DenseNet121** and **ResNet50**, using transfer learning. Furthermore, a hybrid approach is evaluated where a pre-trained CNN is used as a feature extractor for a classical Multi-Layer Perceptron (MLP).

To ensure the system is interpretable, we have integrated **Gradient-weighted Class Activation Mapping (Grad-CAM)** to produce visual heatmaps. These heatmaps highlight the regions in the X-ray image that were most influential in the model's decision-making process, providing a crucial layer of transparency.

## 2. Repository Structure

```
.
├── data_exp.ipynb          # Main Jupyter Notebook with all experiments and analysis.
├── README.md               # This file.
├── .gitignore              # Specifies files and folders to be ignored by Git.
├── report/                 # Contains the final paper in IEEE format.
│   └── report.pdf
└── requirements.txt        # Lists all Python dependencies for the project.
```
*(Note: The `train/` and `test/` data directories are not included in this repository as per standard practice for large datasets. Please see the 'Dataset' section for download instructions.)*

## 3. How to Run the Code

### 3.1. Prerequisites

- Python 3.8+
- An environment with GPU support (e.g., NVIDIA GPU with CUDA) is highly recommended for training the deep learning models.

### 3.2. Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/is-pneumonia-detection-project.git](https://github.com/YourUsername/is-pneumonia-detection-project.git)
cd is-pneumonia-detection-project
```

**2. Set up a Python virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install the required libraries:**
A `requirements.txt` file is provided for easy installation.
```bash
pip install -r requirements.txt
```

**4. Download the Dataset:**
The project uses the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.
- **Download Link:** [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- After downloading, extract the archive. You should have a `chest_xray` folder.
- **IMPORTANT:** Place the `train` and `test` subdirectories from the downloaded dataset directly into the root of the project folder. The final structure should look like this:
  ```
  .
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── test/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── data_exp.ipynb
  └── ... (other project files)
  ```

### 3.3. Running the Experiment

The entire workflow—from data loading and pre-processing to model training, evaluation, and explainability—is contained within the **`data_exp.ipynb`** Jupyter Notebook.

To run the project:
1.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
2.  Open the `data_exp.ipynb` file.
3.  Execute the cells sequentially from top to bottom. The notebook is commented to explain each step of the process.

**Note for Google Colab Users:**
The notebook is also compatible with Google Colab. If using Colab, you will need to:
1.  Upload the dataset to your Google Drive.
2.  Mount your Google Drive using the provided code cells at the beginning of the notebook.
3.  Update the `BASE_DIR` variable to point to the project's directory within your Google Drive.

## 4. Summary of Workflow

The `data_exp.ipynb` notebook implements the following key steps of the KDD process:

1.  **Data Acquisition & Exploration:** Loads the dataset, analyzes the significant class imbalance, and visualizes sample images.
2.  **Data Pre-processing:** Implements a robust data pipeline using `torchvision.transforms`, including resizing, data augmentation (rotation, flipping, color jitter) for the training set, and normalization for all images.
3.  **Model Selection & Training:**
    -   **Approach 1 (End-to-End):** Implements transfer learning and fine-tuning for **DenseNet121** and **ResNet50**. The training loop includes validation, a weighted loss function (`BCEWithLogitsLoss` with `pos_weight`) to handle class imbalance, and a learning rate scheduler.
    -   **Approach 2 (Hybrid):** Uses the trained DenseNet121 as a feature extractor and trains a classical **MLP classifier** on the extracted high-level features.
4.  **Evaluation:** Compares all models on the unseen test set using metrics such as Precision, Recall, and F1-Score. A confusion matrix is generated for the best-performing model.
5.  **Explainability (XAI):** Implements **Grad-CAM** to generate visual heatmaps that provide a justification for the model's predictions on sample test images.

---