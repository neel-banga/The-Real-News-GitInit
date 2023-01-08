from bs4 import BeautifulSoup
import requests
from transformers import pipeline
from readability import Document

class Viewpoints:

    def __init__(self, topic) -> None:
        self.topic = topic
        self.r_site = self.find_viewpoint_page('republican')
        self.d_site = self.find_viewpoint_page('democratic')
        self.r_text = self.get_HTML_file_content(self.r_site)
        self.d_text = self.get_HTML_file_content(self.d_site)
        self.r_summary = self.summarize(self.r_text)
        self.d_summary = self.summarize(self.d_text)

    def summarize(self, text):
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        summary_text = summarizer(text, max_length=500, min_length=5, do_sample=False)[0]['summary_text']
        return summary_text

    def find_viewpoint_page(self, party):
        
        if party == 'republican':
            search_site = ' "republicanviews.org" '
        elif party == 'democratic':
            search_site = ' "democrats.org" '

        search = f'{search_site} {self.topic}'
        url = 'https://www.google.com/search'

        headers = {
            'Accept' : '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
        }
        parameters = {'q': search}

        content = requests.get(url, headers = headers, params = parameters).text
        soup = BeautifulSoup(content, 'html.parser')

        search = soup.find(id = 'search')
        first_link = search.find('a')

        site_link = first_link['href']
        return site_link

    def get_HTML_file_content(self, link):
        page = requests.get(link)
        doc = Document(page.content)
        return doc.summary()


    def return_summaries(self):
        return self.r_summary, self.d_summary


def get_topic():
    topic = input('What is the topic would you like to explore? \n')
    g_view = Viewpoints(topic)
    r_view, d_view = g_view.return_summaries()
    print(f'\n \n The Republican View \n \n {r_view}')
    print(f'\n \n The Democratic View \n \n {d_view}')