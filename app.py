from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the trained Naive Bayes model and TF-IDF vectorizer
nb_classifier = MultinomialNB()
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Load the dataset (assuming you have 'spam.csv' in the same directory)
df = pd.read_csv('spam.csv', encoding='latin-1')
X = df['v2']
y = df['v1'].map({'spam': 1, 'ham': 0})
X_tfidf = tfidf_vectorizer.fit_transform(X)
nb_classifier.fit(X_tfidf, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        message = [request.form['Text']]
        message_tfidf = tfidf_vectorizer.transform(message)
        prediction = nb_classifier.predict(message_tfidf)
        if prediction[0] == 1:
            result = "Spam"
        else:
            result = "Not Spam"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
