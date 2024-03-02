# PDF Text Classifier with Flask

This is a simple web application built with Flask that allows users to upload PDF files and classify their contents into predefined categories using a trained machine learning model.

# Requirements

- Python 3.x
- Flask
- PyPDF2
- NumPy
- NLTK

# Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/your_username/pdf-text-classifier.git
    ```

2. Navigate to the project directory:

    ```bash
    cd pdf-text-classifier
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have trained machine learning models (`model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`) and place them in the root directory.

2. Start the Flask application:

    ```bash
    python app.py
    ```

3. Open your web browser and go to `http://localhost:8080`.

4. Upload a PDF file and click on the classify button to see the predicted categories and their probabilities.

# File Structure

- `app.py`: Main Flask application containing the classification logic.
- `preprocess.py`: Contains preprocessing functions for text data.
- `templates/`: HTML templates for the web application.
- `uploads/`: Folder to store uploaded PDF files.

# Acknowledgments

- This project utilizes the Flask framework for building web applications.
- PDF parsing is done using PyPDF2 library.
- Text preprocessing is performed using NLTK.
- Machine learning models are trained externally and loaded into the application.
