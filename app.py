from flask import Flask, render_template, request, jsonify, flash
import joblib
import os
from sentence_transformers import SentenceTransformer
from fake_news_utils import predict_article, fetch_similar_articles

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# Load models (with error handling)
try:
    model = joblib.load('fake_news_model1.pkl')
    print("✅ Fake news model loaded successfully")
except Exception as e:
    print(f"❌ Error loading fake news model: {e}")
    model = None

try:
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ BERT model loaded successfully")
except Exception as e:
    print(f"❌ Error loading BERT model: {e}")
    bert_model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}
    
    if request.method == 'POST':
        try:
            text = request.form.get('news_text', '').strip()
            
            # Validation
            if not text:
                flash('Please enter some text to analyze.', 'error')
                return render_template('index.html', result={})
            
            if len(text) < 50:
                flash('Please enter at least 50 characters for better analysis.', 'warning')
                return render_template('index.html', result={})
            
            if not model or not bert_model:
                flash('Models are not properly loaded. Please check server logs.', 'error')
                return render_template('index.html', result={})
            
            # Get prediction
            result = predict_article(text, model, bert_model)
            
            # Get similar articles (with error handling)
            try:
                result['similar'] = fetch_similar_articles(text)
            except Exception as e:
                print(f"Warning: Could not fetch similar articles: {e}")
                result['similar'] = []
            
            # Add some additional metrics
            result['text_length'] = len(text)
            result['word_count'] = len(text.split())
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash('An error occurred during analysis. Please try again.', 'error')
            return render_template('index.html', result={})
    
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (useful for future mobile app or integrations)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if len(text) < 50:
            return jsonify({'error': 'Text too short (minimum 50 characters)'}), 400
        
        if not model or not bert_model:
            return jsonify({'error': 'Models not loaded'}), 500
        
        result = predict_article(text, model, bert_model)
        
        try:
            result['similar'] = fetch_similar_articles(text)
        except:
            result['similar'] = []
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'bert_loaded': bert_model is not None
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting NewsGuard AI Fake News Detection Server...")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') != 'production')
    
    # Check if running in production
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        app.run(debug=True, port=5000)
