# 🔢 Handwritten Digit Recognition System

A PyTorch-based Convolutional Neural Network (CNN) for MNIST handwritten digit classification, implementing custom architecture, data augmentation, and regularization techniques.

---

## Table of Contents

1. [About the Dataset](#about-the-dataset)
2. [Project Structure](#project-structure)
3. [Model Architecture](#model-architecture)
4. [Key Features](#key-features)
5. [Execution Pipeline](#execution-pipeline)
6. [Setup & Installation](#setup--installation)
7. [Usage](#usage)
8. [Contributing](#contributing)

---

## About the Dataset

The system uses the **MNIST Dataset**, which consists of:
- **60,000** training images (28×28 grayscale)
- **10,000** test images
- **10 classes** (digits 0–9)

The dataset is automatically downloaded via `torchvision.datasets`. Images are normalized using the dataset's specific mean and standard deviation.

---

## Project Structure

```text
Digit-Recognition/
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── main.ipynb                  # Main notebook (Training, Evaluation, Testing)
├── CNN_best_model.pth          # Saved model weights (generated during training)
└── data/                       # Dataset storage (created automatically)
```

---

## Model Architecture

The model follows a VGG-style design pattern, stacking convolutional layers with batch normalization and GELU activations before pooling.

**Network Flow:**
```
+-------------------------+
| Input Image (1x28x28)   |
+-------------------------+
            |
            v
+-------------------------------------------------------+
| [BLOCK 1: Feature Extraction]                         |
|                                                       |
| Conv2D(32, k=3) -> BatchNorm -> GELU                  |
|          |                                            |
|          v                                            |
| Conv2D(32, k=3) -> BatchNorm -> GELU                  |
|          |                                            |
|          v                                            |
| MaxPool2D(k=2)                                        |
|          |                                            |
|          v                                            |
| Dropout2d(p=0.2)                                      |
+-------------------------------------------------------+
            |
            v (Tensor Shape: 32x14x14)
            |
+-------------------------------------------------------+
| [BLOCK 2: Deeper Patterns]                            |
|                                                       |
| Conv2D(64, k=3) -> BatchNorm -> GELU                  |
|          |                                            |
|          v                                            |
| Conv2D(64, k=3) -> BatchNorm -> GELU                  |
|          |                                            |
|          v                                            |
| MaxPool2D(k=2)                                        |
|          |                                            |
|          v                                            |
| Dropout2d(p=0.25)                                     |
+-------------------------------------------------------+
            |
            v (Tensor Shape: 64x7x7)
            |
+-------------------------------------------------------+
| [BLOCK 3: Abstract Concepts]                          |
| (No pooling here to preserve spatial grid)            |
|                                                       |
| Conv2D(128, k=3) -> BatchNorm -> GELU                 |
|          |                                            |
|          v                                            |
| Dropout2d(p=0.3)                                      |
+-------------------------------------------------------+
            |
            v (Tensor Shape: 128x7x7)
            |
+-------------------------------------------------------+
| [CLASSIFIER HEAD]                                     |
|                                                       |
| Flatten (Input vector size: 128*7*7 = 6272)           |
|          |                                            |
|          v                                            |
| Linear(6272->512) -> BN1d -> GELU -> Dropout(p=0.35)  |
|          |                                            |
|          v                                            |
| Linear(in=512, out=10)                                |
+-------------------------------------------------------+
            |
            v
+-------------------------+
| Final Output (10 Logits)|
+-------------------------+
```

---

## Key Features

-   **Hardware Acceleration:** Supports both **CUDA** (Linux/Windows) and **MPS** (macOS) for GPU acceleration, falling back to CPU if unavailable.
-   **Data Augmentation:** Applies random rotations (±10°), translations (±10%), and scaling (90-110%) during training to improve generalization.
-   **Regularization:** Uses staggered Dropout rates (0.2 to 0.35) and Label Smoothing (0.1) to prevent overfitting.
-   **Evaluation Setup:** Includes vectorized accuracy metrics, confusion matrix generation, and visual error analysis.
-   **Configurable Data Loading:** Centralized settings for batch_size, num_workers, and pin_memory allow users to optimize throughput for their specific hardware. (Note: Default settings are pre-tuned for Apple Silicon M3 Pro efficiency).

---

## Execution Pipeline

The code execution process is automated with the following logic:

1.  **Model Check:** The script checks for existing weights (`CNN_best_model.pth`).
    * If found, training is skipped, and weights are loaded.
    * If not found, the training loop begins.
2.  **Optimization:**
    * **Optimizer:** Adam (`lr=1e-3`)
    * **Scheduler:** StepLR (decays learning rate by 0.5 every 10 epochs)
    * **Early Stopping:** Stops training if validation loss does not improve for 5 consecutive epochs.
3.  **Testing:** The best performing model (lowest validation loss) is loaded for final evaluation on the test set.

---

## Setup & Installation

This project requires Python 3.10 or higher.

### Method 1: Using pip (Standard)

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Method 2: Using uv (Alternative/Recommended)

1. Install uv:
    ```bash
    pip install uv
    ```

2.  Create and activate a virtual environment:
    ```bash
    uv venv --python=python3.10
    source .venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

---

## Usage

1.  **Download Pre-trained Weights (Optional):**
    If you prefer to skip training and use the pre-trained model immediately:
    * Go to the **[Releases](https://github.com/Avaneesh40585/Digit-Recognition/releases)** section of this repository.
    * Download the `CNN_best_model.pth` file.
    * Place it in the **root directory** of the project (`Digit-Recognition/`).
    * *The script will automatically detect this file and load the weights instead of training.*

2.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab main.ipynb
    ```

3.  **Run the Pipeline:**
    Execute the notebook cells. The main execution block handles the logic for training versus loading:

    ```python
    # Automatically handles Training vs Loading based on file existence
    run_pipeline(cnn_model, train_loader, valid_loader, test_loader, device)
    ```

4.  **View Outputs:**
    The notebook will display:
    * Loss and Accuracy plots.
    * A Confusion Matrix heatmap.
    * A grid of misclassified images.

---

## Contributing

I welcome contributions! Whether it's optimizing the CNN architecture, adding new augmentation techniques, or improving the visualization, here is how you can help.

### How to Get Started

1.  **Fork the repository**
    ```bash
    # Clone your fork
    git clone [https://github.com/yourusername/Digit-Recognition.git](https://github.com/yourusername/Digit-Recognition.git)
    cd Digit-Recognition
    ```

2.  **Create a feature branch**
    ```bash
    git checkout -b feature/new-architecture
    ```

3.  **Make your changes**
    * **Notebooks:** Ensure `main.ipynb` runs sequentially without errors (`Restart Kernel and Run All Cells`).
    * **Code Style:** Follow standard Python conventions (PEP 8).
    * **Validation:** If you change the model, please run the full pipeline and include the new test accuracy in your PR description.

4.  **Commit your changes**
    ```bash
    git commit -m "Add elastic distortion to data augmentation"
    ```

5.  **Push to your fork**
    ```bash
    git push origin feature/new-architecture
    ```

6.  **Open a Pull Request**
    * Provide a clear description of what you changed.
    * **Crucial:** If your changes affect model performance, attach a screenshot of the new **Loss/Accuracy plots** or **Confusion Matrix**.
    * Reference any related issues (e.g., `Closes #42`).

### Bug Reports

* Use GitHub Issues with the **"bug"** label.
* Include the specific cell where the error occurred.
* Provide your environment details (OS, PyTorch version, GPU/CPU).

### Feature Requests

* Use GitHub Issues with the **"enhancement"** label.
* Describe the proposed improvement (e.g., "Implement Quantization for mobile deployment").
* Explain the potential benefit to the project.

---

Star the repository if you like it. 🌟!