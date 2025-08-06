from ..utils.common_functions import cosine_similarity
from ..extractors.pptx import PPTExtractor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage

from transformers import AutoTokenizer

# Initialiser les embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_embedding(text):
    return embedding_model.embed_query(text)

def count_tokens(text):
    return len(tokenizer.tokenize(text))

class PPTSummarizer(PPTExtractor):
    def __init__(self, file_path, extraction_method: str = "slide", ocr_engine: str = "tesseract") -> None:
        super().__init__(file_path, extraction_method, ocr_engine)
        super().extract()

    def summarize(self, summarize_method="slide", slide_number=0, summarize_model="mistral-small:22b-instruct-2409-q5_K_M", system_prompt="Tu reçois les informations d'une diapositive PowerPoint, telles que le texte, les tableaux, les graphiques ou le texte extrait d’images (OCR). Génère un résumé concis et précis de la diapositive, en citant les sources utilisées (tableaux/graphiques) si applicable."):
        llm = OllamaLLM(model=summarize_model)

        if summarize_method == "slide":
            summarize_response = []
            for slide in self.slides:
                slide_text = slide.slide_text
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=slide_text)
                ]
                response = llm.invoke(messages)
                summarize_response.append({
                    "Slide Number": slide.slide_number,
                    "Title": slide.slide_title,
                    "Summary": response
                })
            return summarize_response
        else:
            raise Exception("Summarize method not supported.")

    def summarize_stream(self,summarize_method : str = "slide",slide_number : int = 0, summarize_model : str ="mistral-small:22b-instruct-2409-q5_K_M", system_prompt : str = "Tu reçois les informations d'une diapositive PowerPoint, telles que le texte, les tableaux, les graphiques ou le texte extrait d’images (OCR). Tu dois générer un résumé de la diapositive. Assure-toi de citer tes sources si ta réponse provient d’un tableau ou d’un graphique, et sois précis."):
        llm = OllamaLLM(model=summarize_model)

        if summarize_method == "slide":
            for slide in self.slides:
                slide_text = slide.slide_text
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=slide_text)
                ]
                response = llm.invoke(messages)
                yield {
                    "Slide Number": slide.slide_number,
                    "Title": slide.slide_title,
                    "Summary": response
                }

        elif summarize_method == "charts":
            for slide in self.slides:
                for entity in slide.entities:
                    if entity.chart_type in ["table", "chart"]:
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=entity.text)
                        ]
                        response = llm.invoke(messages)
                        yield {
                            "Slide Number": slide.slide_number,
                            "Entity Type": entity.chart_type,
                            "Summary": response
                        }
        else:
            raise Exception("Summarize method not supported.")

class PPTQnA(PPTSummarizer):
    def __init__(self, file_path, extraction_method: str = "slide", ocr_engine: str = "tesseract") -> None:
        super().__init__(file_path, extraction_method, ocr_engine)
        self.retriever = []
        self.prompt = ""

    def run(self, query, method="similarity", model="mistral-small:22b-instruct-2409-q5_K_M", k=10, similarity_score=0.6):
        print("lha9 hna ?!!")
        llm = OllamaLLM(model=model)

        if method == "similarity":
            query_embedding = get_embedding(query)
            for slide in self.slides:
                sim = cosine_similarity(query_embedding, slide.embeddings)
                slide.similarity = sim if sim >= similarity_score else 0

            self.retriever = sorted(self.slides, key=lambda x: x.similarity, reverse=True)
            prompt = "On vous fournit le contenu de diapositives PowerPoint. À partir de ce contenu, essayez de répondre à la question suivante. Soyez clair et précis ; si vous ne connaissez pas la réponse, répondez simplement « Je ne sais pas ». Lors de votre réponse, citez le numéro de la diapositive comme source. Exemple de citation correcte : [1] [2] [3] ; exemple incorrect : [1,2,3].\n Question : " + query + "\nContenu des diapositives :\n\n"

            for slide in self.retriever[0:min(k, len(self.retriever))]:
                prompt += f"Slide [{slide.slide_number}]:  {slide.slide_text}\n"

            self.prompt = prompt
            messages = [
                SystemMessage(content="Vous êtes un assistant utile pour répondre aux questions basées sur le contenu de diapositives PowerPoint."),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            return response

        elif method == "all":
            prompt = "You are provided with PPT slide content. Based on the provided Slide Content try to answer the following question. Be clear and answer accurately, if not answer 'I don't know'. While answering cite the slide number as source. Example (Correct cites: [1] [2] [3], Incorrect: [1,2,3]). \n Question: " + query + "\nSlide Wise Content:\n\n"
            for slide in self.slides:
                prompt += f"Slide [{slide.slide_number}]:  {slide.slide_text}\n"

            self.prompt = prompt
            messages = [
                SystemMessage(content="You are a helpful assistant for answering questions based on PowerPoint slide content."),
                HumanMessage(content=prompt)
            ]
            response = llm.invoke(messages)
            return response

        else:
            raise Exception(f"Method '{method}' not supported.")
