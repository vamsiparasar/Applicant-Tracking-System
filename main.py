from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please run 'python -m spacy download en_core_web_sm' to install the spaCy English model.")

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def calculate_score(job_description, resume):
    job_tokens = preprocess_text(job_description)
    resume_tokens = preprocess_text(resume)
    intersection = len(set(job_tokens).intersection(resume_tokens))
    union = len(set(job_tokens).union(resume_tokens))
    similarity_score = (2 * intersection) / (union + intersection)  # Adjusted for a score in the range of 0 to 1
    scaled_score = round(similarity_score * 100)  # Scale to a score in the range of 1 to 100
    return scaled_score

def evaluate_resume(job_description, resume):
    score = calculate_score(job_description, resume)

    # Get entities from job description and resume
    try:
        job_entities = extract_entities(job_description)
        resume_entities = extract_entities(resume)
    except NameError:
        job_entities = []
        resume_entities = []

    # Provide specific suggestions based on entities
    suggestions = []
    for entity in job_entities:
        if entity not in resume_entities:
            suggestions.append(f"Consider emphasizing your experience in {entity}.")

    return score, suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    job_description = request.form['job_description']
    resume = request.form['resume']
    score, suggestions = evaluate_resume(job_description, resume)

    # Set the CSS class based on the score range
    if score <= 30:
        result_class = 'low-score'
        result_message = 'Your resume may need significant improvement.'
    elif score <= 70:
        result_class = 'medium-score'
        result_message = 'Your resume has potential for improvement.'
    else:
        result_class = 'high-score'
        result_message = 'Your resume is a good match for the job description!'

    return render_template('result.html', score=score, result_class=result_class,
                           result_message=result_message, suggestions=suggestions)

if __name__ == '__main__':
    app.run(debug=True)
