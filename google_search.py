###########################################################################################################################
#  This file searches google via google's API to find the most similar n related results, then extracts the body of these #
#  documents, in addition to other meta information like the author, publish date, and title.                             #
#                                                                                                                         #
#  ______________________________________________________________________________________________________________________ #
#  Author : Israa Qasim Jaradat , IDIR lab, University of Texas at Arlington
#                                                                                                                         #
###########################################################################################################################


from googlesearch import search
from newspaper import Article
from random import randrange
from bert_embeddings import find_most_similar

# tld : tld stands for top level domain which means we want to search our result on google.com or google.in or some other domain.
# lang : lang stands for language.
# num : Number of results we want.
# start : First result to retrieve.
# stop : Last result to retrieve. Use None to keep searching forever.
# pause : Lapse to wait between HTTP requests. Lapse too short may cause Google to block your IP. Keeping significant lapse will make your program slow but its safe and better option.
# Return : Generator (iterator) that yields found URLs. If the stop parameter is None the iterator will loop forever.

class Analyzed_article:
    def __init__(self,text,id):
        self.id = id
        self.text =text
        self.preprocessed_text =""
        self.most_relevant_sent =[]
        self.stance = "UNKNOWN"
        self.author = "UNKNOWN"
        self.publish_date ="UNKNOWN"
        self.summary = "UNKNOWN"
        self.keywords ="UNKNOWN"
        self.image ="UNKNOWN"
        self.url = "UNKNOWN"
        self.year  = "UNKNOWN"
        self.html=""
        self.source="UNKNOWN"

def search_claim(claim):
    print("Searching with Google ...")
    urls = []
    for result in search(claim, tld="com", num=20, stop=20, pause=randrange(5,10), lang='en', start=0):   #   ==>>> using random number generator to generate the value of pause to avoid being blocked by google
        urls.append(result)
    articles =[]
    for url in urls:
        article = Article(url)
        try:
            article.download()
        except:
            print ("Unable to download the article: "+url)
        try:
            article.parse()
            article.full_url = url
            articles.append(article)
        except:
            print ("Unable to parse the article :"+ url)
    print("finished searching with Google ...")
    return articles



def preprocess_article_text(text):
    text= text.replace('\n','. ')
    text= text.replace('\t','')
    return text


def analyze_article(article,claim,n_relevant):
    print ('Analyzing article ...')
    relevant_sentences = find_most_similar(article, claim)
    if len(relevant_sentences) > n_relevant:
        return relevant_sentences[0: n_relevant-1]
    else:
        return None

def do_research(claim):
    print ("Research started ...")
    relevant_articles = search_claim(claim)
    analyzed_articles = []
    i=0
    for a in relevant_articles:
        if a.text != "":
            # links = analyze_urls(a)
            print("Processing article #" + str(i))  # NOTE: the article may or may not be added to the output file based on its length")

            article = Analyzed_article(a.text, i)
            article.preprocessed_text = preprocess_article_text(a.text)
            article.most_relevant_sent = analyze_article(article.preprocessed_text, claim,5) # gets the most relevant 5 sentences from the body of the article
            if article.most_relevant_sent is not None:
                if len(a.authors) > 0:
                    article.author = a.authors
                if a.publish_date is not None:
                    formatted_date = a.publish_date.strftime("%d-%b-%Y")
                    article.publish_date = formatted_date
                    article.year = a.publish_date.year
                if a.top_image != '':
                    article.image = a.top_image
                if a.summary != '':
                    article.summary = a.summary
                if len(a.keywords) > 0:
                    article.keywords = a.keywords
                if a.source_url != '':
                    article.source = a.source_url
                if a.html != '':
                    article.html = a.html
                if a.full_url != '':
                    article.url = a.full_url
                analyzed_articles.append(article)
        i += 1
    return analyzed_articles


