import os
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Imports de IA
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACIÓN ---
# Nota: En Hugging Face es mejor configurar esto como "Secret" en los ajustes del Space
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
app = FastAPI()

# Habilitar CORS para que tu web de Vercel pueda conectar siempre
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

temas_ayuda = (
    "reservas de vuelos interplanetarios, entrenamiento de astronautas, "
    "turismo en el Monte Olimpo de Marte y paquetes de colonización en el Domo Alpha"
)

nuevo_contenido = """
MarsX Horizon es la primera agencia de turismo espacial privada liderada por Daniel Molnar.

Nuestra misión y servicios exclusivos:
- Vuelos Directos: Rutas semanales desde la Tierra hasta el Cráter Gale en naves de última generación.
- Alojamiento de Lujo: Estancias en los Domos Bio-Sostenibles con vista a las lunas Fobos y Deimos.
- Entrenamiento: Programa intensivo de 3 meses para civiles que incluye simulaciones de gravedad cero y supervivencia en entornos áridos.
- Expediciones: Tours guiados al Monte Olimpo (el volcán más alto del sistema solar) y exploración de cuevas de lava.
- Tecnología: Utilizamos propulsión de plasma y sistemas de reciclaje de oxígeno de ciclo cerrado para máxima seguridad.
- Soporte: Asistencia 24/7 para colonos sobre trámites de ciudadanía marciana y logística de suministros.
"""

# --- PREPARACIÓN DE DATOS ---
if not os.path.exists("data"):
    os.makedirs("data")

file_path = "data/mars_knowledge.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(nuevo_contenido.strip())

# --- RAG PIPELINE ---
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Los embeddings se descargarán al iniciar el Space
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists("./mars_vector_db"):
    shutil.rmtree("./mars_vector_db")

vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./mars_vector_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

template = f"""ESTÁS EN MODO 'OFICIAL DE COMUNICACIONES MARSX HORIZON'.
CONTEXTO: {{context}}
REGLAS:
1. Bienvenida épica mencionando: {temas_ayuda}.
2. Respuesta técnica si está en el contexto.
3. Si no es de Marte, decir: "Lo siento, solo informo sobre: {temas_ayuda}."
PREGUNTA: {{question}}
RESPUESTA:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# --- ENDPOINTS ---
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "MarsX API is Running"}

@app.post("/ask")
async def ask_ai(request: ChatRequest):
    try:
        response = rag_chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
