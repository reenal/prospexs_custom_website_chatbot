from langchain.document_loaders import UnstructuredURLLoader

'''
def extract_data_from_webpage(web_url):
    try:
        print('Inside extract_data_from_webpage')
        
        # ! pip3 install unstructured libmagic python-magic python-magic-bin
        loader = UnstructuredURLLoader(urls=[web_url])
        documents = loader.load()
        print('documents - ', documents)
        return documents
    except:
        print('Exception inside extract_data_from_webpage')
        

if __name__ == "__main__":
    URLs = [
    "https://ineuron.ai/"
     ]   
    documents=extract_data_from_webpage(URLs)
    print(documents)
'''

'''
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
webpage = r"chat.openai.com" # edit me
userprompt = input("Enter prompt here:\n")
searchterm = userprompt # edit me

driver = webdriver.Chrome()
driver.get(webpage)

sbox = driver.find_element_by_class_name("textarea")
sbox.send_keys(searchterm)

submit = driver.find_element_by_class_name("button")
submit.click()

text = driver.find_element_by_class_name("p")
print("text")

'''




# ! pip install langchain_community
# ! pip install bs4 -q
site_url = [
    "https://prospexs.ai/"
     ]

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(site_url)
docs = loader.load()
print(docs)