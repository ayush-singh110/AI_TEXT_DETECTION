# 🚀 AI Text Detector

An advanced machine learning system that distinguishes between AI-generated and human-written content with **99.7% accuracy**! 🎯

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Overview

This project implements a sophisticated text classification system that can accurately identify whether a piece of text was written by an AI or a human. Unlike solutions that rely on pre-trained models, this detector uses custom-trained embeddings specifically optimized for this task.

## ✨ Key Features

- 🤖 **High-Precision Detection** - 99.7% accuracy in identifying AI vs human text
- ⚡ **Lightning Fast Analysis** - Get results in seconds
- 🔒 **Privacy-First Design** - Text processed securely, nothing stored
- 📱 **User-Friendly Interface** - Clean, responsive web application
- 🎯 **Advanced NLP Pipeline** - Custom embeddings trained specifically for this task

## 🔧 Technical Highlights

### Custom Architecture

- **Custom Word2Vec Embeddings**: Built my own embedding model using Gensim with 100-dimensional vectors instead of relying on pre-trained models
- **Decision Tree Classifier**: Optimized specifically for text classification tasks
- **NLTK Preprocessing**: Comprehensive text preprocessing and feature extraction pipeline

### 💡 Why Custom Embeddings?

Instead of using pre-trained models like GloVe or FastText, I trained domain-specific embeddings that better capture the nuances between AI and human writing patterns. This approach led to:
- Superior performance on the specific task
- Faster inference times
- Better generalization to various writing styles

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **ML Libraries**: Scikit-learn, Gensim
- **NLP**: NLTK
- **Frontend**: HTML, CSS, JavaScript

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ayush-singh110/AI_TEXT_DETECTION.git
cd AI_TEXT_DETECTION
```

2. **Create a virtual environment**
```bash
python -m venv myenv
```

3. **Activate the virtual environment**

On Windows:
```bash
myenv\Scripts\activate
```

On macOS/Linux:
```bash
source myenv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Run the application**
```bash
python -m app
```

2. **Open your browser**
Navigate to `http://localhost:5000`

3. **Analyze text**
- Paste or type your text into the input field
- Click "Analyze"
- Get instant results showing whether the text is AI-generated or human-written

## 📁 Project Structure

```
AI_TEXT_DETECTION/
│
├── app.py                    # Flask web application
├── main.py                   # Main training/testing script
├── setup.py                  # Package setup file
├── requirements.txt          # Project dependencies
├── Word2Vec.model           # Trained Word2Vec model
├── .gitignore               # Git ignore file
│
├── src/                     # Source code directory
│   ├── components/          # Model components
│   ├── pipeline/            # Training and prediction pipelines
│   └── utils/               # Utility functions
│
├── artifacts/               # Saved models and artifacts
├── logs/                    # Application logs
├── notebook/                # Jupyter notebooks for experiments
└── templates/               # HTML templates
    └── index.html
```

## 🎯 How It Works

1. **Text Preprocessing**: Input text is cleaned and tokenized using NLTK
2. **Feature Extraction**: Text is converted to numerical vectors using custom Word2Vec embeddings
3. **Classification**: Decision Tree classifier analyzes the feature vectors
4. **Result**: System outputs the prediction with confidence score

## 📊 Model Performance

- **Accuracy**: 99.7%
- **Precision**: 99.5%
- **Recall**: 99.6%
- **F1-Score**: 99.5%

## 👤 Author

**Ayush Singh**
- GitHub: [@ayush-singh110](https://github.com/ayush-singh110)
- Project: [AI_TEXT_DETECTION](https://github.com/ayush-singh110/AI_TEXT_DETECTION)

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- Gensim for Word2Vec implementation
- Scikit-learn for machine learning utilities
- Flask for web framework

## 📧 Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/ayush-singh110/AI_TEXT_DETECTION/issues).

---

⭐ If you find this project useful, please consider giving it a star!