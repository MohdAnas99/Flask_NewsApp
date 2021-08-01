import numpy as np
from collections import defaultdict
from gensim import corpora,models,similarities
import gensim
import jinja2
import os
import tempfile
import json
import requests
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

json_srch = ""
documents=[]
@app.route('/form', methods=['GET', 'POST']) #allow both GET and POST requests
def form():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        #language = request.form.get('language')
        #framework = request.form['framework']
        search = request.form['search']
        json_srch = request.form.to_dict()
        #print(json_srch)
        #return jsonify(json_srch)
        #return '''<h1>The language value is: {}</h1>'''.format(search)
        URL = "https://rapidapi.p.rapidapi.com/api/search/NewsSearchAPI"
        HEADERS = {
            "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
            "x-rapidapi-key": "563b896006msh3f0da36c9b16552p111fcejsn71f8fc4a5dd6"
        }

        query = search
        page_number = 1
        page_size = 10
        auto_correct = True
        safe_search = False
        with_thumbnails = True
        from_published_date = ""
        to_published_date = ""

        querystring = {"q": query,
                       "pageNumber": page_number,
                       "pageSize": page_size,
                       "autoCorrect": auto_correct,
                       "safeSearch": safe_search,
                       "withThumbnails": with_thumbnails,
                       "fromPublishedDate": from_published_date,
                       "toPublishedDate": to_published_date}

        response = requests.get(URL, headers=HEADERS, params=querystring).json()

        print(response)

        total_count = response["totalCount"]

        for web_page in response["value"]:
            url = web_page["url"]
            title = web_page["title"]
            description = web_page["description"]
            body = web_page["body"]
            documents.append(web_page["description"])
            date_published = web_page["datePublished"]
            language = web_page["language"]
            is_safe = web_page["isSafe"]
            provider = web_page["provider"]["name"]

            image_url = web_page["image"]["url"]
            image_height = web_page["image"]["height"]
            image_width = web_page["image"]["width"]

            thumbnail = web_page["image"]["thumbnail"]
            thumbnail_height = web_page["image"]["thumbnailHeight"]
            thumbnail_width = web_page["image"]["thumbnailWidth"]

            #print("Url: {}. Title: {}. Published Date: {}.".format(url, title, date_published))


        #return response
#################################################################################################################################
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
        frequency = defaultdict(int)

        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [
            [token for token in text if frequency[token] > 1]
            for text in texts
        ]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

        doc = search

        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow]  # convert the query to LSI space
        #print(vec_lsi)



        with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
            lsi.save(tmp.name)  # same for tfidf, lda, ...

        loaded_lsi_model = models.LsiModel.load(tmp.name)


        index = gensim.similarities.MatrixSimilarity(lsi[corpus])

        index.save('tmp.name')
        index = similarities.MatrixSimilarity.load('tmp.name')


        sims = index[vec_lsi]  # perform a similarity query against the corpus
        #print(list(enumerate(sims)))
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        avg = 0
        n=1

        for doc_position, doc_score in sims:
            n=n+1
            print(doc_score, documents[doc_position])
            #print(doc_score)
            avg = avg + doc_score
            if n == 7:
               break

#################################################################################################################################
        avg = avg/8
        favg = round(avg*100)
        print(favg)
        res = json.dumps(response)
        resp = json.loads(res)
        return render_template('form.html', resp = resp ,favg= favg)
        #return render_template('data.html', resp = resp)


    return render_template('index.html')


app.run(host='localhost', port=4000)
