from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

# Initialiser les embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialiser le tokenizer pour le comptage des tokens
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_embedding(text_to_embed: str):
    """
    Génère les embeddings pour un texte donné à l'aide d'un modèle HuggingFace local.
    
    Args:
        text_to_embed (str): Le texte à encoder.
    
    Returns:
        List[float]: Le vecteur d'embedding.
    """
    return embedding_model.embed_query(text_to_embed)

def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens dans un texte (selon le tokenizer BERT).
    
    Args:
        text (str): Le texte à analyser.
    
    Returns:
        int: Nombre de tokens.
    """
    return len(tokenizer.tokenize(text))