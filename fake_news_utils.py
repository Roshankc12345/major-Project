import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from gnews import GNews
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    logger.info("✅ NLTK stopwords loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading NLTK stopwords: {e}")
    stop_words = set()

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for analysis
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    try:
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        
        # Remove special characters and digits, keep only letters
        text = re.sub(r"[^a-zA-Z]", " ", text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        if stop_words:
            text = ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])
        
        return text
        
    except Exception as e:
        logger.error(f"Error in text cleaning: {e}")
        return text

def fetch_similar_articles(text: str, max_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch similar articles from Google News
    
    Args:
        text (str): Text to search for similar articles
        max_articles (int): Maximum number of articles to return
        
    Returns:
        List[Dict]: List of similar articles
    """
    try:
        gnews = GNews(language='en', max_results=max_articles)
        
        # Extract key keywords from the first 15 words (better than 10)
        words = text.split()[:15]
        keywords = ' '.join([word for word in words if len(word) > 3])
        
        # If no good keywords found, use first 10 words
        if len(keywords.strip()) < 10:
            keywords = ' '.join(text.split()[:10])
        
        logger.info(f"Searching for articles with keywords: {keywords[:50]}...")
        
        results = gnews.get_news(keywords)
        
        # Clean and format results
        formatted_results = []
        for article in results:
            try:
                formatted_article = {
                    'title': article.get('title', 'No title available'),
                    'description': article.get('description', ''),
                    'url': article.get('url', '#'),
                    'published_date': article.get('published date', 'Date not available'),
                    'publisher': article.get('publisher', {})
                }
                formatted_results.append(formatted_article)
            except Exception as e:
                logger.warning(f"Error formatting article: {e}")
                continue
        
        logger.info(f"✅ Found {len(formatted_results)} similar articles")
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Error fetching similar articles: {e}")
        return []

def predict_article(text: str, model, bert_model, threshold: float = 0.6) -> Dict[str, Any]:
    """
    Predict if an article is fake or real
    
    Args:
        text (str): Article text to analyze
        model: Trained classification model
        bert_model: BERT model for embeddings
        threshold (float): Confidence threshold for warnings
        
    Returns:
        Dict: Prediction results
    """
    try:
        # Clean the text
        cleaned = clean_text(text)
        
        if not cleaned or len(cleaned.strip()) < 10:
            return {
                'prediction': 'Unknown',
                'confidence': 0,
                'warning': True,
                'error': 'Text too short or empty after cleaning',
                'probabilities': {'fake': 0.5, 'real': 0.5}
            }
        
        # Get BERT embeddings
        try:
            vector = bert_model.encode([cleaned])
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Get prediction probabilities
        try:
            prob = model.predict_proba(vector)[0]
            label = model.predict(vector)[0]
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            raise
        
        # Format results
        prediction = 'Real' if label == 1 else 'Fake'
        confidence = round(np.max(prob) * 100, 2)
        
        # Add additional analysis
        word_count = len(text.split())
        analysis_quality = 'Good' if word_count > 100 else 'Limited' if word_count > 50 else 'Poor'
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'warning': confidence < (threshold * 100),
            'probabilities': {
                'fake': round(prob[0] * 100, 2),
                'real': round(prob[1] * 100, 2)
            },
            'analysis_quality': analysis_quality,
            'word_count': word_count,
            'cleaned_length': len(cleaned)
        }
        
        logger.info(f"✅ Prediction: {prediction} ({confidence}% confidence)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return {
            'prediction': 'Error',
            'confidence': 0,
            'warning': True,
            'error': str(e),
            'probabilities': {'fake': 0.5, 'real': 0.5}
        }

def get_text_statistics(text: str) -> Dict[str, Any]:
    """
    Get detailed statistics about the text
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict: Text statistics
    """
    try:
        words = text.split()
        sentences = text.split('.')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': round(sum(len(word) for word in words) / len(words), 2) if words else 0,
            'avg_sentence_length': round(len(words) / len([s for s in sentences if s.strip()]), 2) if sentences else 0
        }
    except Exception as e:
        logger.error(f"Error calculating text statistics: {e}")
        return {}