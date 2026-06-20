from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
from langchain_huggingface import HuggingFaceEmbeddings
import os
import traceback

router=APIRouter()

@router.post("/ask/")

async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("USING EMBED MODEL:", type(embed_model))
        print("QUESTION:", question)
        print("STARTING EMBEDDING")

        embedded_query = embed_model.embed_query(question)

        print("EMBEDDING SUCCESS")
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]


        class SimpleRetriever(BaseRetriever):
            docs: List[Document]

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.docs

        # retriever = SimpleRetriever(docs)
        retriever = SimpleRetriever(docs=docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        traceback.print_exc()
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})