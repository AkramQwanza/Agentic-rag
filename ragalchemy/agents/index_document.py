from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

import os
import pandas as pd
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.documents import Document
from ..extractors.pptx import PPTExtractor
from .pptx2 import PPTSummarizer

def index_document(pdf_path: str, Extractor: PPTSummarizer ) -> WeaviateVectorStore:
    """
    Charge un fichier PDF, le découpe, l'encode et l'indexe dans FAISS.
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF à indexer.
        index_path (str): Dossier où sauvegarder l'index FAISS.

    Returns:
        FAISS: L'objet FAISS vectorstore généré.
    """
    print(f"📥 Chargement du document : {pdf_path}")

    # Initialiser les embeddings
    #✅ SEMANTIC CHUNKING
    print("\n🧠 Découpage sémantique intelligent...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
   
    documents = []
    for slide in Extractor.slides:
        print("Slide number:", slide.slide_number)
        try:
            doc = Document(
                page_content=slide.slide_text,
                metadata={
                    "source": Extractor.file_path,
                    "slide_title": slide.slide_title,
                    "page": slide.slide_number,
                    "type": "powerpoint"
                }
            )
            documents.append(doc)
        except Exception as e:
            print(f"❌ Erreur sur slide {slide.slide_number} : {e}")

    print(f"✅ {len(documents)} documents creer.")

    # Ne pas découper les documents PowerPoint qui sont déjà découpés par slide
    chunks = []
    for doc in documents:
        if doc.metadata.get('type') == 'powerpoint':
            chunks.append(doc)
        else:
            chunks.extend(text_splitter.split_documents([doc]))

    print(f"✅ {len(chunks)} chunks créés.")

    # ✅ CRÉATION DU VECTOR STORE AVEC WEAVIATE
    print("\n💾 Enregistrement dans Weaviate...")
    import weaviate
    import weaviate.classes.config as wvcc
    # client = weaviate.Client("http://localhost:8080")  # Assurez-vous que Weaviate tourne
    print(weaviate.__version__)

    # ✅ CONNEXION WEAVIATE V4
    print("\n💾 Connexion à Weaviate...")
    client = weaviate.connect_to_local(skip_init_checks=True)

    # ✅ CRÉATION DE L'INDEX AVEC SYNTAXE V4
    try:
        if not client.collections.exists("test"):
            collection = client.collections.create(
                name="test",
                properties=[
                    wvcc.Property(
                        name="content",
                        data_type=wvcc.DataType.TEXT
                    ),
                    wvcc.Property(
                        name="source",
                        data_type=wvcc.DataType.TEXT
                    ),
                    wvcc.Property(
                        name="page",
                        data_type=wvcc.DataType.NUMBER
                    )
                ]
            )
        print("✅ Collection Weaviate prête")
    except Exception as e:
        print(f"❌ Erreur création collection: {str(e)}")
        client.close()
        exit(1)

    # Nettoyage des métadonnées des chunks
    for chunk in chunks:
        # On ne garde que les métadonnées essentielles
        cleaned_metadata = {
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", 0)
        }
        chunk.metadata = cleaned_metadata

    # UTILISATION AVEC LANGCHAIN
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="test",
        text_key="content",
        embedding=embeddings,
        attributes=["source", "page"]  # Spécifier les attributs à stocker
    )

    try:
        print("\n💾 Ajout des documents dans Weaviate...")
        vectorstore.add_documents(chunks)
        print(f"✅ {len(chunks)} chunks ajoutés avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout des documents : {str(e)}")
        client.close()
        exit(1)

    # print(f"📊 Nombre d'objets dans la collection : {client.collections.get('qwanza_docs').count_objects()}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(chunk.page_content)
        print("="*40)

    # print("✅ Index Weaviate créé.")

    # collection = client.collections.get("qwanza_docs")
    # results = collection.query.fetch_objects(limit=10)

    # for i, obj in enumerate(results.objects):
    #     print(f"Objet {i}:")
    #     print(obj.properties)

    # print(vars(chunks[0]))
    # print(chunks[0].page_content)
    # print(chunks[0].metadata)

    return vectorstore

if __name__ == "__main__":
    try:
        index_document("AXEREAL.pptx")
    except Exception as e:
        print(f"Erreur : {e}")