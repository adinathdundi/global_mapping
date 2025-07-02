from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")    

snomed_refset = pd.read_csv("refset_extended_map (1).xlsx")
diag = pd.read_csv("diagnosis.csv")

merged_df = pd.merge(
    snomed_refset,
    diag[['CodeWithSeparator', 'LongDescription']],
    how='left',
    left_on='mapTarget',
    right_on='CodeWithSeparator'
)

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i,row in merged_df.iterrows():
        try:
            icd_name = str(row["LongDescription"])
            term = str(row["term"])
            snomed_code = str(row["referencedComponentId"])
            icd_code = str(row["mapTarget"])

            document = Document(
                page_content=term,
                metadata={
                    "mapAdvice": str(row.get("mapAdvice", "")),
                    "snomed_code": snomed_code,
                    "icd_name": icd_name,
                    "icd_code": icd_code
                },
                id=str(i)
            )
        
            ids.append(str(i))
            documents.append(document)
        except:
            print("No mapping found")

    vector_store = Chroma(
        collection_name="snomed_mapping",
        persist_directory=db_location,
        embedding_function=embeddings
    )    
    vector_store.add_documents(documents=documents, ids=ids)

else:
    vector_store = Chroma(
        collection_name="snomed_mapping",
        persist_directory=db_location,
        embedding_function=embeddings
    )


retriever = vector_store.as_retriever(
    search_kwargs={"k":1}
    
)
