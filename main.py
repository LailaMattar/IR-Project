import csv

import pandas as pd
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
import glob
import re
from collections import defaultdict
import pickle
from flask import request
from flask import jsonify, current_app


def split_dataset1():
    df = pd.read_csv('C:/Users/User/Desktop/wikIR1k/documents.csv')
    for i, row in df.iterrows():
        text = row['text_right']
        filename = f"{row['id_right']}.txt"
        with open(f"{filename}", "w") as f:
            f.write(text)
        # Create a folder to store the documents
    if not os.path.exists('documents'):
        os.makedirs('documents')
    # Loop through the rows of the DataFrame and save each document as a separate file
    for i, row in df.iterrows():
        # Extract the relevant information for each document
        text = row['text_right']
        # Save the document as a file with the name of its id
        filename = os.path.join('documents', f"{i}.txt")
        with open(filename, "w") as f:
            f.write(text)
def split_dataset2():
     collection_path = r"C:\Users\User\Desktop\antique\collection.tsv"
     output_dir = r"C:\Users\User\PycharmProjects\IRproject\document2"
     os.makedirs(output_dir, exist_ok=True)
     with open(collection_path, "r", encoding="utf-8") as file:collection_data = file.readlines()
     for line in collection_data:
         doc_id, text = line.strip().split("\t")
         filename = f"{doc_id}.txt"
         with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as file:file.write(text)
"""""
def rename_datase1():

    # Set the paths of the CSV file and the target directory
    csv_file_path = 'C:/Users/User/Desktop/wikIR1k/documents.csv'
    target_directory = 'C:/Users/User/PycharmProjects/IRproject/documents'

    # Read the CSV file and rename the files
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row if present

        for row in csv_reader:
            id_right = row[0]  # Assuming the first column is id_right

            # Get the list of files in the target directory
            file_names = os.listdir(target_directory)

            # Iterate over the files and rename themf"{i}.txt
            for file_name in file_names:
                # Construct the old file path and new file path
                old_file_path = os.path.join(target_directory, file_name)
                new_file_path = os.path.join(target_directory, id_right + '.txt' )

                # Rename the file
                os.rename(old_file_path, new_file_path)

    print('Files renamed successfully!')
def rename_datase2():
    #enter dirctory path
    # Read the CSV file into a DataFrame
    df = pd.read_csv('doc.csv')

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        file_name = row['id_right']
        old_file_path = os.path.join("",
                                     file_name)  # Assuming folder_path is the path to the folder containing the files
        new_file_path = os.path.join("", file_name + '.txt')  # Assuming the files have a ".txt" extension

        # Rename the file
        os.rename(old_file_path, new_file_path)

    print("Files renamed successfully!")
    """

abbreviations = {
    'abbr.': 'abbreviation',
    'etc.': 'etcetera',
    'i.e.': 'that is',
    'e.g.': 'for example',
    'Mr.': 'Mister',
    'Mrs.': 'Missus',
    'DR.': 'Doctor',
    'Eng.': 'Engineer',
    'USA': 'United States of America',
    'UK': 'United Kingdom',
    'mar': 'March',
    'jan': 'January',
    'feb': 'February',
    'apr': 'April',
    'jun': 'June',
    'jul': 'July',
    'dec': 'December',
    'nov': 'November',
    'oct': 'October',
    'sep': 'September',
    'aug': 'August'

    # Add more abbreviations and their full forms as needed
}
def tokenizeing(dir_path):
    inverted_index = defaultdict(list)
    data = defaultdict(list)
    file_extension = '*.txt'
    file_pattern = os.path.join(dir_path, file_extension)
    #for file_path in glob.glob(file_pattern):
    for x in range(0, 10):
        file_path = "C:/Users/User/PycharmProjects/IRproject/documents/{}.txt".format(x)
        print('Processing file:', file_path)
        with open(file_path, "r") as file:
            text = file.read()
            print(text)
            print("---------------------------------------------------------------------")

            # spelling
            spell = Speller(lang='en')
            CorrectedSpell = " ".join([spell(word) for word in text.split()])
            print("Corrected Spell")
            print(CorrectedSpell)

            #normalization abbreviations
            for abbreviation, full_form in abbreviations.items():
                text = CorrectedSpell.replace(abbreviation, full_form)
            print("abbreviations")
            print(text)

            #normalization date
            REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
            REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
            REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
            REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
            REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
            REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
            REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
            COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                                REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

            all_dates = re.findall(COMBINATION_REGEX, text)
            for s in all_dates:
                new_date = parse(s[0]).strftime("%d-%m-%Y")
                text = text.replace(s[0], new_date)
            print("normalized date")
            print(text)

            # remove punctuation
            no_punc_string = re.sub(r'[^\w\s]', '', text)
            print("remove punctuation")
            print(no_punc_string)

            # to lowercase
            lower_string = no_punc_string.lower()
            print("lower case")
            print(lower_string)

            # remove whitespace
            no_wspace_string = lower_string.strip()
            print("white space")
            print(no_wspace_string)

            # tokenizing
            tokens = word_tokenize(no_wspace_string)
            print("tokenizing")
            print(tokens)

            # remove stop word
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            print("stop word")
            print(filtered_tokens)

            # stemming
            stemmer = PorterStemmer()
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
            print("stemming")
            print(stemmed_tokens)

            # lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
            print("lemmatization")
            print(lemmatized_tokens)

            term_count=len(lemmatized_tokens)
            for term in lemmatized_tokens:
                #inverted_index[term].append(file_path)
                inverted_index[term].append(x)
            #data[term]=lemmatized_tokens
            data[x]=lemmatized_tokens

            # join filtered tokens
            preprocessed_text = ' '.join(lemmatized_tokens)
            print("join")
            print(preprocessed_text)

            # Tokenize the preprocessed text and print the tokens
            preprocessed_tokens = word_tokenize(preprocessed_text)
            print("toknize")
            print(preprocessed_tokens)

            print("datasss")
            print(data)
    print("ll")
    print(inverted_index.items())
    return data,inverted_index

def TfidfVectorizer1(data1):
    print(data1)
    docs1 = [' '.join(data1) for index, data1 in data1.items()]
    #docs1 = [' '.join(data1) for data1 in data1]

    # إنشاء مثيل من TfidfVectorizer
    vectorizer1 = TfidfVectorizer()
    tfidf_matrix1 = vectorizer1.fit_transform(docs1)

    with open("tfidf_matrix1.pkl", "wb") as file:
        pickle.dump(tfidf_matrix1, file)
        file.close()

    with open("vectorizer1.pkl", "wb") as file:
        pickle.dump(vectorizer1, file)
        file.close()



    # طباعة مصفوفة الفيكتور
    print(tfidf_matrix1.toarray())

def TfidfVectorizer2(data2):
    data2 = defaultdict(list)
    data = defaultdict(list)
    dataf=defaultdict(list)
    docs2=defaultdict(list)

    docs2 = [' '.join(data2) for index, data1 in data2.items()]
    #docs2 = [' '.join(data1) for data1 in data1]
    """""
    with open("docs1.pickle", 'rb') as file:
        data = pickle.load(file)
        print("jo")
        for doc_id, doc_text in data.items():
             data2[doc_id]= doc_text
             docs2[doc_id].extend([''.join(data2[id]) for id in data2.keys()])
             print(docs2)

             #print(data2[id])
             #docs2.append([''.join(data2[id]) for index, data2[id] in data2.items()])
        print(docs2)
        """""

    # إنشاء مثيل من TfidfVectorizer
    vectorizer2 = TfidfVectorizer()
    tfidf_matrix2 = vectorizer2.fit_transform(docs2)

    with open("tfidf_matrix2.pkl", "wb") as file:
        pickle.dump(tfidf_matrix2, file)

    with open("vectorizer2.pkl", "wb") as file:
        pickle.dump(vectorizer2, file)
        file.close()

    # طباعة مصفوفة الفيكتور
    print(tfidf_matrix2.toarray())

def Inverted_index1(inverted_index1):
    with open("inverted_index1.pkl", "wb") as file:
        pickle.dump(inverted_index1, file)
    print("inverted index")
    print(inverted_index1.items())

def Inverted_index2(inverted_index2):
    with open("inverted_index2.pkl", "wb") as file:
        pickle.dump(inverted_index2, file)
    print("inverted index")
    print(inverted_index2.items())

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/api/data', methods=['GET'])
def get_data():
    # Logic to retrieve and process data
    data = {'key': 'value'}
    return jsonify(data)


@app.route('/api', methods=['POST'])
def process_data():
    # Logic to process the data
    data = request.get_json()  # Assuming the request body contains JSON data
    # Perform operations on the data
    name = data['name']
    age = data['age']

    # Process the data
    return jsonify({'message': 'Data received successfully'})
    result = f"Hello {name}, you are {age} years old!"
    print(result)



@app.route('/search',methods=['POST'])
def search():
    print("search")
    #data=request.get_json()
    #Query=data['Query']
    #DataSetNum=data['DataSetNum']
    Query="mediterranean food"
    DataSetNum="1"

    if DataSetNum=="1":
        print("dsd")
        print(DataSetNum)
        with open("tfidf_matrix1.pkl", "rb") as file:
            tfidf_matrix1 = pickle.load(file)
        tfidf_matrix = tfidf_matrix1
        files=glob.glob('C:/Users/User/PycharmProjects/IRproject/documents' + '/*.txt')
        print("files")
        #print(files)
    else:
        print("kkd")
        print(DataSetNum)
        #with open("tfidf_matrix2.pkl", "rb") as file:
           # returned_tfidf_matrix2 = pickle.load(file)
        #tfidf_matrix = returned_tfidf_matrix2
        #change to dataset2
        #files = glob.glob('C:/Users/User/PycharmProjects/IRproject/documents' + '/*.txt')

    print(tfidf_matrix)

    QueryTokens=word_tokenize(Query)
    spell=Speller(lang='en')
    CorrectedQuery=" ".join([spell(word) for word in Query.split()])
    print("Original Query=",Query)
    print("Corrected Query",CorrectedQuery)

    # normalization abbreviations
    for abbreviation, full_form in abbreviations.items():
        text = CorrectedQuery.replace(abbreviation, full_form)
    print("abbreviations")
    print(text)

    # normalization date
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                        REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    all_dates = re.findall(COMBINATION_REGEX, text)
    for s in all_dates:
        new_date = parse(s[0]).strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)
    print("normalized date")
    print(text)

    #out put for handle function
    tokens = word_tokenize(text)
    print("tokens :",tokens)

    lower=[w.lower() for w in tokens]
    print("lower :",lower)

    stop_words=set(stopwords.words('english'))
    stop = [token for token in lower if token not in stop_words]
    print("stop :",stop)

    ps = PorterStemmer()
    stem = [ps.stem(token) for token in stop]
    print(stem)

    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(token,pos='v') for token in stem]
    print(lem)

    query=' '.join(lem)
    print(query)

    if DataSetNum=="1":
        print("11")
        with open("vectorizer1.pkl", "rb") as file:
            vectorizer1 = pickle.load(file)
        QueryVector = vectorizer1.transform([query])
        print(QueryVector)
    else:
        #QueryVector =returned_vectorizer2.transform([query])
        print("mm")

    CosineSimilarities=cosine_similarity(QueryVector, tfidf_matrix)
    print("CosineSimilarities :")
    print(CosineSimilarities)

    ranking = [file for _, file in sorted(zip(CosineSimilarities[0], files), reverse=True)]
    print("ranking")
    print(ranking)
    with open("ranking1.pkl", "wb") as file:
        pickle.dump(ranking, file)


@app.route('/result' , methods=['GET'])
def result():
    print("hff")
    data = {}
    file_contents = []
    file_names = []

    #ranking = ['C:/Users/User/PycharmProjects/IRproject/documents/8.txt',
              # 'C:/Users/User/PycharmProjects/IRproject/documents/4.txt',
              # 'C:/Users/User/PycharmProjects/IRproject/documents/5.txt']
    with open("ranking1.pkl", "rb") as file:
        ranking1 = pickle.load(file)
    # or file instedof file
    for file_path in ranking1:
        with open(file_path, 'r') as file:
            content = file.read()
            print("The most relevant files are at the top, and the relatedness decreases downwards")
            print("file name")
            print(file_path)
            print("content")
            print(content)
            file_contents.append(content)
            filenames = os.path.basename(file_path)
            file_names.append(filenames)
    data['content'] = file_contents
    data['file_paths'] = file_names
    return jsonify(data)


def load_queries1(query_file):

    queries = {}
    with open(query_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            query_id, query = row['id_left'], row['text_left']
            queries[query_id] = query
    return queries
def evaluation1(queries):
    #data set 1
    for query_id, query_text in queries.items():
        # Perform operations on each query
        print("Query ID:", query_id)
        print("Query Text:", query_text)
        """""
       # search_evaluation
        #  هون صار معي الملفات يلي رجعتهم هي الكويري
        #يلي هنن محتوى الدويكومنت يلي بجيبو من المسار وid الدوكيومنت يلي بجيبو من اسم الملف

        # Read the QREL file into a DataFrame
        qrel_df = pd.read_csv("qrel_file.csv")

        # Iterate over the rows of the DataFrame
        for index, row in qrel_df.iterrows():
            query_id = row["id_left"]
            document_id = row["id_right"]
            relevance_label = row["label"]

            # Perform operations using the query ID, document ID, and relevance label
            print("Query ID:", query_id)
            print("Document ID:", document_id)
            print("Relevance Label:", relevance_label)
            # Add your code here to process each row of the QREL file

    # Load queries
    #استدعاء
    queries = load_queries('query_file.csv')
    """

def searchevalouation(index,Query):
    print("search")
    with open("tfidf_matrix2.pkl", "rb") as file:
        tfidf_matrix2 = pickle.load(file)
    tfidf_matrix = tfidf_matrix2
    files = glob.glob('C:/Users/User/PycharmProjects/IRproject/document2' + '/*.txt')
    print("files")
    # print(files)
    print(tfidf_matrix)

    QueryTokens = word_tokenize(Query)
    spell = Speller(lang='en')
    CorrectedQuery = " ".join([spell(word) for word in Query.split()])
    print("Original Query=", Query)
    print("Corrected Query", CorrectedQuery)

    # normalization abbreviations
    for abbreviation, full_form in abbreviations.items():
        text = CorrectedQuery.replace(abbreviation, full_form)
    print("abbreviations")
    print(text)

    # normalization date
    REGEX_1 = r"(([12]\d|30|31|0?[1-9])[/-](0?[1-9]|1[0-2])[/.-](\d{4}|\d{2}))"
    REGEX_2 = r"((0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9])[/.-](\d{4}|\d{2}))"
    REGEX_3 = r"((\d{4}|\d{2})[/-](0?[1-9]|1[0-2])[/-]([12]\d|30|31|0?[1-9]))"
    REGEX_4 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]),? (\d{4}|\d{2}))"
    REGEX_5 = r"(([12]\d|30|31|0?[1-9]) (January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    REGEX_6 = r"((\d{4}|\d{2}) ,?(January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec) ([12]\d|30|31|0?[1-9]))"
    REGEX_7 = r"((January|February|Mars|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Jun|Jul|Agu|Sept|Sep|Oct|Nov|Dec),? (\d{4}|\d{2}))"
    COMBINATION_REGEX = "(" + REGEX_1 + "|" + REGEX_2 + "|" + REGEX_3 + "|" + \
                        REGEX_4 + "|" + REGEX_5 + "|" + REGEX_6 + ")"

    all_dates = re.findall(COMBINATION_REGEX, text)
    for s in all_dates:
        new_date = parse(s[0]).strftime("%d-%m-%Y")
        text = text.replace(s[0], new_date)
    print("normalized date")
    print(text)

    # out put for handle function
    tokens = word_tokenize(text)
    print("tokens :", tokens)

    lower = [w.lower() for w in tokens]
    print("lower :", lower)

    stop_words = set(stopwords.words('english'))
    stop = [token for token in lower if token not in stop_words]
    print("stop :", stop)

    ps = PorterStemmer()
    stem = [ps.stem(token) for token in stop]
    print(stem)

    lemmatizer = WordNetLemmatizer()
    lem = [lemmatizer.lemmatize(token, pos='v') for token in stem]
    print(lem)

    query = ' '.join(lem)
    print(query)

    print("11")
    with open("vectorizer2.pkl", "rb") as file:
        vectorizer2 = pickle.load(file)

    QueryVector = vectorizer2.transform([query])
    print(QueryVector)

    CosineSimilarities = cosine_similarity(QueryVector, tfidf_matrix)
    print("CosineSimilarities :")
    print(CosineSimilarities)

    ranking = [file for _, file in sorted(zip(CosineSimilarities[0], files), reverse=True)]
    print("ranking")
    print(ranking)
    return ranking

def load_queries2():
    queries2={}
    query_file_path = "C:/Users/User/Desktop/antique/train/queries.txt"  # Path to the query file

    # Open the query file
    with open(query_file_path, 'r') as file:
        # Read each line of the file
        for line in file:
            # Split the line using a tab delimiter
            index, query = line.strip().split('\t')
            queries2[index] = query

            # Process the index and query as needed
            #print("Index:", index)
            #print("Query:", query)
            # Perform other operations with the index and query
    return queries2

def qrel(index):
    print("qrel")
    qrels={}
    with open("C:/Users/User/Desktop/antique/qrels(1).txt", 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            if len(values) != 4:
                continue
            query_id,_, document_id, relevance_label = values
            relevance_label = int(relevance_label)
            query_id=int(query_id)
            document_idd = tuple(document_id.split('\t'))
            if query_id in qrels:
                print("yes")
                qrels[query_id][document_idd] = relevance_label
            else:
               qrels[query_id] = {document_idd: relevance_label}
    print("qrels")
    print(qrels)
    return qrels

    """""
            #qrel_data[query_id]=(document_id,relevance_label)
            qrel_data.setdefault(query_id, []).append((document_id))
        print(qrel_data)
            #qrel_data.append((query_id, document_id, relevance_label))
    if index in qrel_data:
        doc_ids = zip(*qrel_data[index])
    for document_id in doc_ids:
        print(doc_ids)
        """


    #return qrel_data

    """
        for line in file:
            if line.count('\t') >= 3:
                query_id, query_type, document_id, relevance_label = line.strip().split('\t')
                if query_id not in qrel_data:
                 qrel_data[query_id] = {}
                qrel_data[query_id][document_id] = int(relevance_label)
        #if query_id == query_index:
        doc_ids.append(document_id)
        relv = qrel_data.get(query_id, {})

    print(relv)
    print(doc_ids)
    """""
    """
        for line in file:
            if line.count('\t') >= 3:
                query_id, query_type, document_id, relevance_label = line.strip().split('\t')
            if query_id==query_index:
                #if query_id not in qrel_data:
                    #qrel_data[query_id] = {}
                print("yes")
                print(query_id)
                print(query_index)
                count =count+1
                #qrel_data[query_id][document_id] = int(relevance_label)
                qrel_data.append(document_id)
                print("Query ID:", query_id)
                print("Document ID:", document_id)
                print("Relevance Label:", relevance_label)
                print(qrel_data)
                print(count)
            count=0

    print("uu")
    """

def evalouatiion2(queries2):
    print("===========================")
    qrels={}
    relv={}
    #qrel_data = qrel()


    for index, query in queries2.items():
        # Perform operations on each query
        print("Query ID:",index)
        print("Query Text:", query)
        #ranking=searchevalouation(index,query)
        qrels = qrel(index)
        relv=qrels.get(index,{})
        print("qrel data")
        print(qrels)

        #relv=qrel_data.get(index,{})
        print("relv")
        print(relv)

def query_refinement(user_query, dataset):
    # Preprocess the dataset
    stopwords = set(nltk_stopwords.words('english'))
    corpus = []
    for document in dataset:
        tokens = word_tokenize(document.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords]
        preprocessed_document = ' '.join(filtered_tokens)
        corpus.append(preprocessed_document)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Preprocess the user's query
    query_tokens = word_tokenize(user_query.lower())
    filtered_query_tokens = [token for token in query_tokens if token.isalpha() and token not in stopwords]
    preprocessed_query = ' '.join(filtered_query_tokens)

    # Transform the preprocessed query using the same vectorizer
    query_vector = vectorizer.transform([preprocessed_query])

    # Compute cosine similarity between the query vector and document vectors
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort the documents based on cosine similarity in descending order
    document_indices = cosine_similarities.argsort()[::-1]

    # Refine the query using terms from the most similar document
    most_similar_document_index = document_indices[0]
    refined_query = corpus[most_similar_document_index]

    # Get suggestions by extracting terms from the most similar document
    suggested_terms = vectorizer.inverse_transform(tfidf_matrix[most_similar_document_index])

    return refined_query, suggested_terms[0]
# Example usage

user_query = "information retrieval"


def main():
    ##file_path = "C:/Users/User/PycharmProjects/IRproject/documents/{}.txt"
    dir_path1 = "C:/Users/User/PycharmProjects/IRproject/documents"
    #dir_path2 = "C:/Users/User/PycharmProjects/IRproject/document2"
    #split_dataset1()
    #dir_path2=""
    #data1,inverted_index1=tokenizeing(dir_path1)
    #print("data")
    #print(data1)
    #TfidfVectorizer1(data1)
    #returned_vectorizer11=returned_vectorizer1
    #rename_datase1()
    #data2,inverted_index2=tokenizeing(dir_path2)
    #TfidfVectorizer2()
    #Inverted_index1(inverted_index1)
    #Inverted_index2(inverted_index2)

    app.run()
    search()
    #result()



    #sara()
    #split_dataset2()
    #query=load_queries2()
    #print("jhhj")
    #evalouatiion2(query)

    dataset = [
        "This document contains information about retrieval techniques.",
        "The process of retrieving information is crucial for data analysis.",
        "Different methods can be used for information retrieval.",
    ]
    for x in range(1, 4):
        file_path = "C:\\Users\\User\\PycharmProjects\\IRproject\\documents\\{}.txt".format(x)

    refined_query, suggested_terms = query_refinement("world war", file_path)

    print("Refined query:", refined_query)
    print("Suggested terms:", suggested_terms)



if __name__ == "__main__":
    main()








