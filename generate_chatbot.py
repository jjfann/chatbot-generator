import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_selection import chi2,SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import joblib


global intent_queries
global curr_intent
global intents
global filename
global model_name

intent_queries = []
intents = []
filename = ''
model_name = ''
curr_intent = 0

class add_new_intents(tk.Toplevel):
    global intent_queries
    global intents

    def __init__(self, master = None):
         
        super().__init__(master= master)
        self.title('Add New Intents')
        self.label = tk.Label(self,text="New Intents:")
        self.text = tk.Text(self,width=80,height=30)
        self.button = tk.Button(self,text='Add Intents',command=self.add_intents)
        self.label.grid(row=0,column=0,sticky=tk.N)
        self.text.grid(row=0,column=1,columnspan=2,padx=5,pady=5)
        self.button.grid(row=1,column=1,sticky=tk.E,padx=5,pady=5)
    
    def add_intents(self):
        input = self.text.get(1.0,'end-1c')
        for intent in input.split('\n'):
            list_box.insert(tk.END,intent)
            intent_queries.append('')
            intents.append(intent)

class enter_filename(tk.Toplevel):


    def __init__(self, master = None):
        super().__init__(master= master)
        self.title('Enter Filename')
        self.label = tk.Label(self,text = 'Enter Filename:')
        self.entry = tk.Entry(self)
        self.button_a = tk.Button(self,text = 'Load CSV',command = self.load_filename)
        self.button_b = tk.Button(self,text = 'Save CSV',command = self.save_filename)
        self.label.grid(row=0,column=0)
        self.entry.grid(row=0,column=1,columnspan=2,padx=5,pady=5)
        self.button_a.grid(row=1,column=1)
        self.button_b.grid(row=1,column=2)

    def load_filename(self):
        global filename
        filename = self.entry.get()
        self.destroy()
        load_csv()
    
    def save_filename(self):
        global filename
        filename = self.entry.get()
        self.destroy()
        save_csv()

class enter_model_name(tk.Toplevel):

    def __init__(self, master = None):
        super().__init__(master= master)
        self.title('Enter Filename')
        self.label = tk.Label(self,text = 'Enter Model Name:')
        self.entry = tk.Entry(self)
        self.button_b = tk.Button(self,text = 'Save Model',command = self.save_model_name)
        self.label.grid(row=0,column=0)
        self.entry.grid(row=0,column=1,columnspan=2,padx=5,pady=5)
        self.button_b.grid(row=1,column=2)
    
    def save_model_name(self):
        global model_name
        model_name = self.entry.get()
        self.destroy()
        generate_model()

class LemmatiseTokens(object):
    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()
    def __call__(self, query):
        return [self.lemmatiser.lemmatize(word) for word in word_tokenize(query)]

def selected_intent():
    global intent_queries
    global curr_intent
    intent_queries[curr_intent] = text_1.get(1.0,'end-1c')
    curr_intent = list_box.curselection()[0]
    text_1.delete(1.0,tk.END)
    text_1.insert(1.0,intent_queries[list_box.curselection()[0]])

def delete():
    global intents
    global curr_intent
    highlighted = list_box.curselection()[0]
    del intent_queries[highlighted]
    list_box.delete(highlighted)
    del intents[highlighted]
    if highlighted == curr_intent:
        text_1.delete(1.0,tk.END)

def save_csv():
    global intents
    global filename
    global intent_queries
    queries = []
    labels = []
    for i,intent in enumerate(intent_queries):
        for query in intent.split('\n'):
            queries.append(query)
            labels.append(intents[i])
    d = {'Queries':queries,'Labels':labels}
    df = pd.DataFrame(d)
    df.to_csv(filename,index=False,header=False)
    root.title(filename)

def load_csv():
    global intent_queries
    global intents
    global curr_intent
    global filename
    intent_queries = []
    list_box.delete(0,tk.END)
    df = pd.read_csv(filename,names=['Queries','Labels'])
    root.title(filename)
    queries = df['Queries'].to_list()
    labels = df['Labels'].to_list()
    temp = set()
    intents = [x for x in labels if not (x in temp or temp.add(x))]
    for intent in intents:
        list_box.insert(tk.END,intent)
        intent_queries.append('')
    for i,query in enumerate(queries):
        intent_queries[intents.index(labels[i])] += query + '\n'
    curr_intent = 0
    text_1.delete(1.0,tk.END)
    text_1.insert(1.0,intent_queries[curr_intent])

def vectorise_queries(queries,labels):
    vectoriser = Pipeline([
        ('CV', CountVectorizer(tokenizer = LemmatiseTokens())),
        ('CS', SelectPercentile(chi2,percentile=98)),
        ('Tf', TfidfTransformer())
    ])
    vector_queries = vectoriser.fit_transform(queries,labels)
    return vector_queries,vectoriser

def blending(queries,labels):
    models = [
        ('LR',LogisticRegression(C = 10, max_iter=1000)),
        ('SVC', SVC(C=1,kernel='linear',probability=True)),
        ('RF', RandomForestClassifier(max_features='log2',n_estimators=1000,random_state=42)),
        ('NB', MultinomialNB())
    ]
    x_train,x_val,y_train,y_val = train_test_split(queries,labels,stratify=labels,random_state=42,test_size=0.5)
    accs = []
    for i,model in enumerate(models):
        model[1].fit(x_train,y_train)
        accs.append(accuracy_score(y_val,model[1].predict(x_val)))
    print(accs)
    estimators = []
    for i,model in enumerate(models):
        if accs[i] >= 0.85*max(accs):
            estimators.append(model)
    blend = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(C = 10, max_iter=10000),cv='prefit')
    blend.fit(x_val,y_val)
    for estimator in blend.estimators:
        estimator[1].fit(queries,labels)
    return blend

def generate_model():
    global intent_queries
    global model_name
    queries = []
    labels = []
    for i,intent in enumerate(intent_queries):
        for query in intent.split('\n'):
            queries.append(query)
            labels.append(i)
    vector_queries,vectoriser = vectorise_queries(queries,labels)
    model = blending(vector_queries,labels)
    joblib.dump(vectoriser,model_name+'_vectoriser')
    joblib.dump(model,model_name)
    np.savetxt(model_name+'_intents',intents,fmt='%s')

root = tk.Tk()

root.title('Chat Bot Trainer')

label_1 = tk.Label(root,text = 'Intents:')
label_2 = tk.Label(root,text = 'Queries:')

list_box = tk.Listbox(root,height=20,width=50)

text_1 = tk.Text(root,height=20,width=100)

button_1 = tk.Button(root,text='Add New Intents')
button_2 = tk.Button(root,text='Select Intent',command=selected_intent)
button_3 = tk.Button(root,text='Delete Intent',command=delete)
button_4 = tk.Button(root,text='Save/Load CSV')
button_5 = tk.Button(root,text='Generate Model')


label_1.grid(row=0,column=0,sticky=tk.N)

list_box.grid(row=0,column=1,columnspan=3,padx=5,pady=5)
list_box.bind('<Double-1>',lambda e: selected_intent())
button_1.bind("<Button>",lambda e: add_new_intents(root))
button_1.grid(row=1,column=1,padx=5,pady=5)

button_2.grid(row=1,column=2,padx=5,pady=5)

button_3.grid(row=1,column=3,padx=5,pady=5)

label_2.grid(row=0,column=4,sticky=tk.N)

text_1.grid(row=0,column=5,columnspan = 3,padx=5,pady=5)

button_4.grid(row=1,column=6,padx=5,pady=5)
button_4.bind("<Button>",lambda e: enter_filename(root))

button_5.grid(row=1,column=7,padx=5,pady=5)
button_5.bind("<Button>",lambda e: enter_model_name(root))

root.mainloop()