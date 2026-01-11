# IMDB Movie Review Sentiment Analysis

A simple web application that predicts whether a movie review is **Positive** or **Negative** using a basic Recurrent Neural Network (RNN) trained on the classic IMDB dataset.

**🚀 [Live Demo](https://cosmicveil1-movie-review-main-ytdikc.streamlit.app/)** | **📊 [GitHub Repo](https://github.com/cosmicveil1/movie-review)**

## 🎯 Key Features

- ✅ **Real-time Sentiment Analysis** - Binary classification (Positive/Negative) with confidence scores
- ✅ **Deployment** - Live application deployed on Streamlit Cloud
- ✅ **Pre-trained Deep Learning Model** - SimpleRNN trained on 25,000 IMDB reviews
- ✅ **Text Preprocessing Pipeline** - Automatic tokenization, encoding, and padding

## 🏗️ Architecture & Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, TensorFlow/Keras |
| **Model** | SimpleRNN (Recurrent Neural Network) |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Data Processing** | NumPy, Scikit-learn |
| **Visualization** | Matplotlib, TensorBoard |

## 📊 Model Performance

- **Dataset**: IMDB Reviews (25,000 training samples)
- **Architecture**: Embedding Layer → SimpleRNN → Dense Output
- **Input Length**: 500 tokens (padded/truncated)
- **Output**: Binary classification with confidence score (0-1)
- **Threshold**: 0.5 for positive/negative classification

## 🚀 Quick Start

### Try the Live App
Click the link above to test the app instantly without installation!

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/cosmicveil1/movie-review.git
cd movie-review
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run locally**
```bash
streamlit run main.py
```

5. **Open browser**
```
http://localhost:8501
```

## 📝 How It Works

### Text Processing Pipeline
```
User Input 
    ↓
Lowercase & Tokenize
    ↓
Map to IMDB Indices
    ↓
Pad/Truncate to 500 tokens
    ↓
RNN Model Inference
    ↓
Confidence Score & Sentiment Label
```

### Example Usage
**Input**: *"This movie absolutely blew my mind! Best cinematography I've ever seen."*

**Output**: 
- Sentiment: **Positive** ✅
- Confidence: **0.9847**

## 📂 Project Structure

```
movie-review/
├── main.py                    # Streamlit web application
├── simpleRNN.h5              # Pre-trained neural network model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── simpleRNN.ipynb          # Model training notebook
├── embedding.ipynb          # Embedding layer analysis
└── prediction.ipynb         # Prediction examples & validation
```

## 🔧 Requirements

```
tensorflow
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras
```

## 📚 Key Implementation Details

### Model Architecture
- **Embedding Layer**: Maps word indices to 128-dimensional vectors
- **SimpleRNN Layer**: 128 units for sequential text processing
- **Dense Output**: Binary classification with sigmoid activation
- **Total Parameters**: ~1.3 million
### Preprocessing
- Uses IMDB's pre-built word index (10,000 most common words)
- Unknown words mapped to index 2
- Sequences padded/truncated to 500 tokens
- Vocabulary size: 10,000 + 3 reserved indices

## 🌐 Deployment

Deployed on **Streamlit Cloud** for instant, serverless hosting:
- Automatic deployment from GitHub
- Real-time updates from repository
- Free tier with unlimited apps
- HTTPS enabled by default

**Deploy your own:**
1. Fork/clone this repo
2. Push to your GitHub account
3. Visit [share.streamlit.io](https://share.streamlit.io)
4. Connect your repo and deploy!

## 📈 Future Enhancements

- [ ] Multi-class sentiment (negative, neutral, positive)
- [ ] Attention mechanism visualization
- [ ] LSTM/GRU architecture comparison
- [ ] Model interpretability with LIME
- [ ] Batch prediction API
- [ ] Performance metrics dashboard
- [ ] Fine-tuning capability
- [ ] Multi-language support

## 💡 Learning Outcomes

This project demonstrates:
- ✓ Building and training RNN models with Keras
- ✓ Text preprocessing and NLP pipelines
- ✓ Model persistence and loading
- ✓ Creating web apps with Streamlit
- ✓ End-to-end ML project workflow

## 📄 License

MIT License - feel free to use this for personal or commercial projects. See [LICENSE](LICENSE) file.


## 👤 Author

**cosmicveil1** - [GitHub](https://github.com/cosmicveil1) 


