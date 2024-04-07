# %%
import os

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith("."):  # Skip hidden files
                file_path = root + "/" + file
                file_list.append(file_path)
    return file_list

# # Specify the directory path
# directory_path = "Archive/Fearless Yachts"

# # Get the list of file paths
# file_paths = list_files(directory_path)


# %%


# %%
def concatenate_texts(texts, max_overlap=0):
    result = []
    prev_text = ""

    for text in texts:
        overlap = 0
        for i in range(min(len(prev_text), len(text), len(prev_text))):
            if prev_text[-i-1:] == text[:i+1]:
                overlap = i + 1
                if overlap >= max_overlap:
                    break

        result.append(text[overlap:])
        prev_text = text

    return "".join(result)





# # %%
# from pdf_mining import segment_documents
# import fitz  # PyMuPDF

# doc = fitz.open(pdf_path)
# metadata = doc.metadata


# toc = doc.get_toc()
# print(toc)

# %%
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document



def naive_page_loading(pdf_path):
    
    def extract_text_from_pdf(pdf_path):
        reader = PdfReader(pdf_path)
        texts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append({"page_number": i + 1, "text": text})
        return texts


    # Extract text from the PDF along with page numbers
    texts = extract_text_from_pdf(pdf_path)

    # Create Langchain Documents with page numbers as metadata
    documents = [Document(page_content=t["text"], metadata={"page_number": t["page_number"]}) for t in texts]

    return documents





# documents = naive_page_loading(pdf_path)
# print(documents)

# Split the documents into chunks using Langchain's text splitter
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)

# Create an embedding using Hugging Face's all-MiniLM-L6-v2 model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



# %%
# pdf_path



# doc = documents[0]

from pdf_mining import split_further2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def inject_metadata(document, metadict):
    metadata=document.metadata
    metadata.update(metadict)
    return Document(page_content=document.page_content,
                    metadata=metadata)


# splitted = split_further2(documents, threshold= 128)

# final_splitted = [inject_metadata(doc, {'source': pdf_path}) for doc in splitted]





# # %%


from langchain.docstore.document import Document

def break_pages(documents):
    if not documents:
        return None

    page_contents = [doc.page_content for doc in documents]
    concatenated_content = '\n'.join(page_contents)

    metadata = documents[0].metadata

    return [Document(page_content=concatenated_content, metadata=metadata)]

def single_pdf_text_handler(pdf_path, breaking_pages = False, splitter_args = {}):
    pages_docs= naive_page_loading(pdf_path)
    
    if breaking_pages:
        pages_docs = break_pages(pages_docs)
    splitted = split_further2(pages_docs, **splitter_args)

    final_splitted = [inject_metadata(doc, {'source': pdf_path, 'in_file': i}) for i, doc in enumerate(splitted)]
  

    return final_splitted
    
    

# all_pdf_docs = []

# for file_path in file_paths:
#     all_pdf_docs+=single_pdf_text_handler(file_path)

# print(all_pdf_docs[:20])




import os

def list_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith("."):  # Skip hidden files
                file_path = root + "/" + file
                file_list.append(file_path)
    return file_list

# # Specify the directory path

# # Get the list of file paths
# file_paths = list_files(directory_path)

# # Print the file paths
# for file_path in file_paths:
#     print(file_path)
    
def get_all_docs(directory, handler_args = {}):
    file_list = list_files(directory)
    all_pdf_docs = []
    for file_path in file_list:
        all_pdf_docs+=single_pdf_text_handler(file_path,**handler_args)
    return all_pdf_docs



all_pdf_docs = get_all_docs("Archive/Fearless Yachts")

def search_objects(directory):
    
    all_pdf_docs = get_all_docs(directory)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store from the chunks and embeddings
    vdb = FAISS.from_documents(all_pdf_docs, embeddings)
    
    return all_pdf_docs, vdb



def get_vdb(all_pdf_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store from the chunks and embeddings
    vdb = FAISS.from_documents(all_pdf_docs, embeddings)
    
    return vdb


# # %%
# all_pdf_docs, vdb = search_objects(directory_path)

# # %%
# all_pdf_docs[0].metadata

# # %%
# search_result = vdb.similarity_search("Gary accused")

# # %%
# extracted_relevant_pages = naive_page_loading(search_result[0].metadata['source'])

# # %%
# search_result

# %%
def get_neighbors(docs, target, key = 'page_number', key_r = 5):
    result = []
    for doc in docs:
        
        if target - key_r <= doc.metadata[key] <= target + key_r:
            result.append(doc)
            
    return result




# get_neighbors(all_pdf_docs, search_result[0].metadata['page_number'])

def get_neighbors_from_docs(docs, target_doc, key = 'page_number', key_r = 5):
    result = []
    metadata = target_doc.metadata
    for doc in docs:
        
        if metadata[key] - key_r <= doc.metadata[key] <= metadata[key] + key_r and metadata['source']==doc.metadata['source']:
            result.append(doc)
            
    return result






def get_neighbors_string_from_docs(docs, target_doc, n=32, overlap_length=32, key='in_file'):
    metadata = target_doc.metadata
    source = metadata['source']
    target_in_file = metadata[key]

    prev_words = []
    next_words = []

    k = 1
    while len(prev_words) < n:
        prev_block = None
        for doc in docs:
            if doc.metadata['source'] == source and doc.metadata[key] == target_in_file - k:
                prev_block = doc
                break

        if prev_block:
            prev_block_words = prev_block.page_content.split()
            if overlap_length > 0:
                prev_block_words = prev_block_words[-min(len(prev_block_words), n - len(prev_words) + overlap_length):]
                prev_block_words = prev_block_words[:-overlap_length]
            else:
                prev_block_words = prev_block_words[-min(len(prev_block_words), n - len(prev_words)):]
            prev_words = prev_block_words + prev_words
        else:
            break

        k += 1

    k = 1
    while len(next_words) < n:
        next_block = None
        for doc in docs:
            if doc.metadata['source'] == source and doc.metadata[key] == target_in_file + k:
                next_block = doc
                break

        if next_block:
            next_words.extend(next_block.page_content.split())
        else:
            break

        k += 1

    prev_words = prev_words[-n:]
    next_words = next_words[:n]

    result = prev_words + [target_doc.page_content] + next_words
    return ' '.join(result)



# get_neighbors_from_docs(all_pdf_docs, search_result[0], key = 'in_file')#, key_r = 10)

# %%
import pandas as pd

def documents_to_dataframe(documents):
    data = []
    for doc in documents:
        row = {'page_content': doc.page_content, **doc.metadata}
        data.append(row)

    df = pd.DataFrame(data)
    return df

df = documents_to_dataframe(all_pdf_docs)
df

# %%
from langchain.docstore.document import Document

def dataframe_to_documents(df):
    documents = []
    for _, row in df.iterrows():
        metadata = dict(row.drop('page_content'))
        doc = Document(page_content=row['page_content'], metadata=metadata)
        documents.append(doc)

    return documents


# documents = dataframe_to_documents(df)
# documents

# %%
from langchain.docstore.document import Document

def find_longest_shared_ngram(s1, s2):
    # Split s1 and s2 into words
    words_s1 = s1.split()
    words_s2 = s2.split()
    
    # Initialize variables to keep track of the largest shared n-gram
    max_length = 0
    max_ngram = ""

    # Iterate through each word in s1
    for i, word in enumerate(words_s1):
        # Break if there are not enough words left in s1 for an n-gram
        if i + max_length >= len(words_s1):
            break

        # Initialize variables for comparison
        found_count = 0
        match_count = 0
        j = 0
        while j < len(words_s2):
            # Check if the word matches the current word of s1
            if word == words_s2[j]:
                found_count += 1
                # Check if the following words in both strings are the same
                k = 1
                jumper = None
                match_count = 1
                while i + k < len(words_s1) and j + k < len(words_s2) and words_s1[i + k] == words_s2[j + k]:
                    if jumper is None:
                        if words_s1[i] == words_s2[i+k]:
                            jumper = k
                    k += 1
                    match_count += 1
                
                # Update max_ngram if a longer shared n-gram is found
                if match_count > max_length:
                    max_length = match_count
                    max_ngram = " ".join(words_s1[i:i + k])
                # Jump to the next potential match
                if jumper is not None:
                    j += jumper
                else:
                    j += k
            else:
                # Jump to the next potential match
                j += 1
   # Return the largest shared n-gram found
    return max_ngram

def hard_similarity_search(query,documents, r,
                           lower = False):
    results = []
    query_length = len(query.split())

    for doc in documents:
        content = doc.page_content
        if lower:
            largest_ngram_length = len(find_longest_shared_ngram(query.lower(), content.lower()).split())
        else:
            largest_ngram_length = len(find_longest_shared_ngram(query, content).split())
        ratio = largest_ngram_length / query_length

        if ratio >= r:
            results.append((doc, largest_ngram_length))

    results.sort(key=lambda x: x[1], reverse=True)
    sorted_documents = [doc for doc, _ in results]

    return sorted_documents



# query = "final decision"
# r = 0.5

# similar_docs = hard_similarity_search(all_pdf_docs, query, r, lower = True)

# similar_docs[:10]

# # %% [markdown]
# 




# %%


# %%
# !pip install llama-cpp-python --force-reinstall --quiet





# %%

def collate_neighbors(docs):
    texts = [doc.page_content for doc in docs]
    return concatenate_texts(texts, max_overlap = 64)


# print(
#     collate_neighbors(

# get_neighbors_from_docs(all_pdf_docs, search_result[3], key = 'page_number', key_r = 10)

# )
# )






# %%
def search_metadata_util(query, docs, search_func, search_args={},
                         neighbor_func = get_neighbors_from_docs,n_args = {},
                         collate_func = collate_neighbors):
    search_doc_lists = search_func(query, **search_args)
    
    ns = []
    for doc in search_doc_lists:
        
        for nei in neighbor_func(docs, doc, **n_args):
            if nei not in ns:
                ns.append(nei)

    return collate_func(ns)

# search_metadata_util("Gary fears", all_pdf_docs, search_func = vdb.similarity_search, search_args = {'k':5},
#                      n_args = {'key': 'in_file', 'key_r': 20},
# )
    

def search_metadata_util2(query, docs, search_func, search_args={},
                         n_args = {'n':32, 'overlap_length':32, 'key':'in_file'},
                         collate_func = collate_neighbors,
                         sep='...\n...'):
    search_doc_lists = search_func(query, **search_args)
    
    ns = ''
    for doc in search_doc_lists:
        
        ns+=sep+get_neighbors_string_from_docs(docs, doc, **n_args)#, key_r = 10)

    return ns

# search_metadata_util("Gary fears", all_pdf_docs, search_func = vdb.similarity_search, search_args = {'k':5},
#                      n_args = {'key': 'in_file', 'key_r': 20},
# )
    

# %%
from functools import partial
from langchain.tools import Tool



def get_neighboring_text(doc, key, key_r):
    # Extract the value of the specified key from the document's metadata
    key_value = doc.metadata.get(key)

    # Retrieve the neighboring documents based on the key value and range
    start = max(0, key_value - key_r)
    end = key_value + key_r
    neighboring_docs = [d for d in all_pdf_docs if start <= d.metadata.get(key) <= end]

    # Concatenate the content of the neighboring documents
    neighboring_text = ' '.join([d.page_content for d in neighboring_docs])

    return neighboring_text



# %%


# %%
# search_tool_func("final judgement")

# %%
# !pip install langchain_anthropic

# %%
# from private_llms.claude import claude as llm

# %%
