from langchain_text_splitters import HTMLSectionSplitter

from bs4 import BeautifulSoup
import requests

def scrape_web(url, file_path):
    res = requests.get(url)
    content = res.content

    soup = BeautifulSoup(content, features="lxml")
    paragraphs = soup.find_all('p')
    
    f = open(file_path, "w")
    f.write(str(paragraphs))

def text_splitter(file_path):
    size = 500
    overlap = 30

    split_headers = [
        ("h1", "Header1"),
        ("h2", "Header2"),
        ("h3", "Header3"),
        ("p", "Paragraph")
    ]
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    section_splitter = HTMLSectionSplitter(split_headers)
    section_splits = section_splitter.split_text(content)

    return section_splits
    # f = open("processed.txt", "w")
    # f.write(str(splits))