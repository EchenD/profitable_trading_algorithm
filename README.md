## Project Title: Deep Learning-Based Cryptocurrency Trend Forecasting

This repository contains a test and study implementation of a deep learning-based model for cryptocurrency trend forecasting, inspired by the research published in the following paper:

> [Cryptocurrency trend forecasting using a hybrid deep learning strategy](https://www.sciencedirect.com/science/article/pii/S0957417423023084) – Expert Systems with Applications, 2023

### Disclaimer

This project was developed solely for academic and research testing purposes. It is not intended for production use or financial decision-making.

---

## Features

* Cryptocurrency historical data preprocessing
* Technical indicators with `ta` and `TA-Lib`
* Feature scaling and class balancing
* Deep learning model using TensorFlow/Keras
* Model evaluation with classification metrics
* Visualization with Matplotlib and Seaborn
* SHAP-based feature explainability

---

## Project Structure

```
.
├── data/                 # Folder for raw and processed datasets
├── models/               # Saved Keras models
├── notebooks/            # Jupyter notebooks (optional)
├── scripts/              # Python scripts for training and evaluation
├── utils/                # Utility modules (e.g., preprocessing, indicators)
├── README.md             # Project overview and instructions
├── requirements.txt      # Python dependencies
└── main.py               # Main training and evaluation pipeline
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main script

```bash
python main.py
```

---

## Requirements

See `requirements.txt` for a full list of dependencies. Main packages include:

* TensorFlow
* scikit-learn
* imbalanced-learn
* shap
* ta
* TA-Lib
* pandas, numpy
* matplotlib, seaborn

---

## Results

This project is for experimentation and reproduces core ideas from the original research paper. For performance and results, refer to your local training and evaluation logs.

---

## License

This repository is released under the MIT License.

---


## Contact

For any questions or suggestions, feel free to open an issue.
