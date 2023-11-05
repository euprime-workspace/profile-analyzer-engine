import os
import locale

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms.ctransformers import CTransformers
from starlette.middleware.cors import CORSMiddleware


# Setting up locale in case not available automatically
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding

FILE_CHUNK_SIZE = 1024

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=512)

qa_template = """Use the following pieces of information to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=qa_template, input_variables=['context', 'question'])
chain_type_kwargs = {"prompt": prompt}
qa_chain = None
llm = None

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionModel(BaseModel):
    question: str


class AnswerModel(BaseModel):
    answer: str


@app.post("/upload/")
def upload(file: UploadFile):
    try:
        global qa_chain
        global llm
        with open(file.filename, 'wb') as tempPDF:
            while contents := file.file.read(FILE_CHUNK_SIZE * FILE_CHUNK_SIZE):
                tempPDF.write(contents)
            pdf_loader = PyPDFLoader(tempPDF.name)
        pdf_pages = pdf_loader.load_and_split()
        texts = text_splitter.split_documents(pdf_pages)
        db = Chroma.from_documents(texts, embeddings)
        llm = CTransformers(
            model='llama-2-7b-chat.ggmlv3.q2_K.bin',
            model_type='llama',
            config={
                'max_new_tokens': 256,
                'temperature': 0.01
            }
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False,
            chain_type_kwargs=chain_type_kwargs,
        )
        os.remove(file.filename)
    except Exception as e:
        return JSONResponse(f"Error: {e}", status_code=500)
    finally:
        tempPDF.close()
    return JSONResponse(f"Extracted context from: {file.filename} and LLM has been initialized for Question Answering")


@app.post("/ask/", response_model=AnswerModel)
async def ask(request: QuestionModel):
    global llm
    if not llm:
        return AnswerModel(answer='LLM initialization in process, please wait')
    global qa_chain
    if not qa_chain:
        return AnswerModel(answer='Extracting context, please try again')
    output = qa_chain(request.question)
    return AnswerModel(answer=output['result'])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
