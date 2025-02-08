from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

# Initialize Flask App
app = Flask(__name__, template_folder="templates")
api = Api(app)

# Load FAISS vector store
print("Loading FAISS vector store...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("FAISS vector store loaded successfully!")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question using the provided context. If the answer is not in the context, say "Answer not found in the context." Be clear and concise.
    
    Context:
    {context}

    Question: {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# API Resource for Chatbot
class ChatBot(Resource):
    def get(self):
        return jsonify({"message": "Use POST request with a question!"})  # ✅ Handle GET requests

    def post(self):
        data = request.get_json()
        user_question = data.get("question", "").strip()
        if not user_question:
            return jsonify({"error": "No question provided!"})
        
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        return jsonify({"response": response["output_text"]})

# Add API endpoint
api.add_resource(ChatBot, "/chat")

# Route for the Webpage UI
@app.route("/")
def home():
    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
