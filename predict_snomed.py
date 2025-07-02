
def lookup_icd_description(icd_code,merged_df) :
        matches = merged_df[merged_df["mapTarget"].str.strip().str.upper() == icd_code.strip().upper()]
        if not matches.empty:
            return matches.iloc[0]["LongDescription"]
        else:
            return None

def lookup_snomed_code(selected_term,merged_df):
        matches = merged_df[merged_df["term"].str.strip().str.lower() == selected_term.strip().lower()]
        if not matches.empty:
            return matches.iloc[0]["referencedComponentId"]
        else:
            return "SNOMED code not found"

def best_matching_snomed(icd_code):
    
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from vector import retriever
    import pandas as pd

    model = OllamaLLM(model = "llama3.2")

    template = """
    You are a medical coding assistant. Given the ICD code and disease description, and a list of possible SNOMED terms, choose the SNOMED term that best matches the disease.

    ICD Description: {icd_description}

    SNOMED Term Candidates:
    {term_list}

    Only return the SNOMED term, exactly as it appears above. Do not generate any codes.
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    snomed_refset = pd.read_csv("refset_extended_map (1).xlsx")
    diag = pd.read_csv("diagnosis.csv")

    merged_df = pd.merge(
        snomed_refset,
        diag[['CodeWithSeparator', 'LongDescription']],
        how='left',
        left_on='mapTarget',
        right_on='CodeWithSeparator'
    )

    icd_description = lookup_icd_description(icd_code,merged_df)
    if not icd_description:
        print("❌ ICD code not found in dataset.")
        return "(No match found)"

    print(f"ℹ️ ICD Description: {icd_description}")
    retrieved_docs = retriever.invoke(icd_description)

    terms = [doc.page_content for doc in retrieved_docs]
    term_list = "\n".join([f"- {term}" for term in terms])  

    # candidates = "\n".join([
    #     f"- {doc.page_content} (SNOMED: {doc.metadata.get('snomed_code')})"
    #     for doc in retrieved_docs
    # ])

        
    result = chain.invoke({"icd_description" : icd_description,"term_list" : term_list}).strip()

    snomed_code = lookup_snomed_code(result,merged_df)
    print(result)
    print(snomed_code)
    return snomed_code