from flask import Flask, request, render_template
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

app = Flask(__name__)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([f"{doc.page_content} - {doc.metadata.get('id', 'Unknown')}" for doc, _score in results])
        
        # Paraphrased response
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        paraphrased_prompt = prompt_template.format(context=context_text, question=query_text)
        model = Ollama(model="mistral")
        paraphrased_response = model.invoke(paraphrased_prompt)
        
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_paraphrased_response = f"{paraphrased_response}\n\n<i>Sources: {', '.join(sources)}</i>"
        
        return formatted_paraphrased_response, context_text
    except Exception as e:
        return f"An error occurred: {e}", ""

@app.route('/', methods=['GET', 'POST'])
def index():
    paraphrased_response = ""
    exact_response = ""
    if request.method == 'POST':
        query_text = request.form['query_text']
        paraphrased_response, exact_response = query_rag(query_text)
    return render_template('index.html', paraphrased_response=paraphrased_response, exact_response=exact_response)

if __name__ == '__main__':
    app.run(debug=True)
