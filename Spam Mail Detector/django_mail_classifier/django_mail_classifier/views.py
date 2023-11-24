from django.shortcuts import render
from django.http import HttpResponse
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open(r'C:/Users/asus/Desktop/Projects/Machine Learning Concepts/Spam Mail Detector/vectorizer.pkl','rb'))

model = pickle.load(open(r'C:/Users/asus/Desktop/Projects/Machine Learning Concepts/Spam Mail Detector/model.pkl','rb'))

def homePage(request):
    return render(request,'index.html')

def predictPage(request):
        msg = ''
        try:
                if request.method=="POST":
                        msg = request.POST.get('msg')
                        print(msg)
                        transformed_msg = transform_text(msg)
                        print(transformed_msg)
                        X = tfidf.transform([transformed_msg])
                        X_new = X.toarray()
                        prediction = model.predict(X_new)[0]
                else:
                        pass
        except:
                return HttpResponse('Error')
                        
        result = ''
        if prediction == 1:
                result = 'Spam'
        elif prediction == 0:
                result = 'Ham'
        else:
                return HttpResponse('Error')
                
        results = {
                'output':'This Message is ' + result + " !",
        }
                        
        return render(request,'predictionPage.html', results)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    new_text = " ".join(y)
    
    return new_text