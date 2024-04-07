
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def replace_newlines(text):
    """
    Replace '\newpage' with '\n\n', and replace any non-consecutive '\n' with a space,
    while preserving consecutive '\n' instances.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text with newline replacements.
    """
    # Replace '\newpage' with '\n\n'
    text = text.replace('\\newpage', '\n\n')

    # Replace non-consecutive '\n' with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    return text

def split_further2(doc, threshold = 32,
                   chunk_overlap = 8,
                   length_function = lambda x: len(x.split(' '))
                   ):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=threshold,
        chunk_overlap=chunk_overlap,
        separators=[' '],  # Split on periods, question marks, exclamation marks, and double newlines
        length_function=length_function,
        )
    splitted = text_splitter.split_documents(doc)
    return splitted
    

get_largest_hierachical_order = lambda text: max([text_i[0] for text_i in text])
def get_sorted_indexes(text, l):
    sorted_indexes = []
    for entry in text:
        if entry[0] == l:
            sorted_indexes.append(entry[2])
    return sorted(sorted_indexes)






def segment_documents(toc, pages, l):
    segmented_documents = []

    # Iterate through the Table of Contents
    
    l_max = get_largest_hierachical_order(toc)
    
    i = 0
    
    len_toc = len(toc)
    
    
    seg_list = []
    current_metadata = {}
    while i< len_toc:
        level, title, index = toc[i]
        if level == l:
            if i != 0:
                if 'start' not in current_metadata.keys():
                    current_metadata['start'] = 0
                    if 'desc' not in current_metadata.keys():
                        current_metadata['desc'] = [] 
                current_metadata['end']=index
                seg_list.append(current_metadata)
            
            current_metadata = {"toc": toc[i],
                                "start": index,
                                "desc": []}
        elif level>l:
            current_metadata['desc'] += toc[i]
        i+=1
    current_metadata['end'] = len(pages)-1
    seg_list.append(current_metadata)
    
    print(seg_list)
    seg_docs = []
    for seg in seg_list:
        page_contents = [page.page_content for page in pages[seg["start"] : seg["end"]+1] ]
        seg_docs.append(Document(page_content='\n\\newpage'.join(page_contents),
                              metadata = seg))
    return seg_docs






