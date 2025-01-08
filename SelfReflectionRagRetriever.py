import os
from typing import List
from langchain.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class DocumentRetriever:
    def __init__(self, model_dir: str, persist_dir: str = "./chroma_db"):
        self.model_dir = model_dir
        self.persist_dir = persist_dir
        self.embedding_model = self._initialize_embeddings()

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model with CPU configuration"""
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'device': 'cpu', 'batch_size': 32}

        return HuggingFaceEmbeddings(
            # model_name=self.model_dir,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def load_documents(self, doc_directory: str) -> List:
        """Load documents from the specified directory"""
        docs_list = []

        for filename in os.listdir(doc_directory):
            file_path = os.path.join(doc_directory, filename)

            if filename.endswith('.txt'):
                loader = TextLoader(file_path)
                docs_list.extend(loader.load())
            elif filename.endswith(('.docx', '.doc')):
                loader = UnstructuredWordDocumentLoader(file_path)
                docs_list.extend(loader.load())

        return docs_list

    def split_documents(self, docs_list: List) -> List:
        """Split documents into chunks"""
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=2500, chunk_overlap=0
        # )
        text_splitter = RecursiveCharacterTextSplitter(
            # separators=["Question:", "Answer:"],
            chunk_size=1024,  # Larger to fit full QA pairs
            chunk_overlap=50,  # Some overlap to catch context
            length_function=len
        )
        return text_splitter.split_documents(docs_list)



    def create_vectorstore(self, doc_splits: List) -> Chroma:
        # First create the vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding= self.embedding_model,
            persist_directory="./chroma_db"
        )

        # Persist
        vectorstore.persist()
        return vectorstore

    def get_retriever(self):
        """Get retriever from existing vectorstore"""
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name="rag-chroma",

        )
        return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "score_threshold": 0.7
        }
        )


def main():
    # Configuration
    model_dir = "/home/ali/moradi/models/Radman-Llama-3.2-3B/extra"
    doc_directory = "/home/ali/moradi/Conference_Content"

    # Initialize retriever
    retriever = DocumentRetriever(model_dir)

    # Process documents and create vectorstore
    docs_list = retriever.load_documents(doc_directory)
    doc_splits = retriever.split_documents(docs_list)
    vectorstore = retriever.create_vectorstore(doc_splits)

    print("Database created and persisted successfully!")


if __name__ == "__main__":
    main()