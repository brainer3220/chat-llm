import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt6.QtGui import QFont, QMovie
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# Set up the API key
os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY"

# Load multiple and process documents
loader = DirectoryLoader('./talk', glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create Chroma DB
persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)

# Make a retriever
retriever = vectordb.as_retriever()

# Make a chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

# Worker thread for processing LLM response
class LLMProcessingThread(QThread):
    response_ready = pyqtSignal(dict)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        llm_response = qa_chain(self.query)
        self.response_ready.emit(llm_response)

# Create the main application window
class ChatBotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatBot GUI")
        self.setGeometry(200, 200, 500, 500)
        self.setup_ui()

    def setup_ui(self):
        # Create the central widget and the layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create the chat history text area
        self.chat_history_text = QTextEdit()
        self.chat_history_text.setReadOnly(True)
        self.chat_history_text.setFont(QFont("Arial", 14))
        self.chat_history_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        layout.addWidget(self.chat_history_text)

        # Create the query input field
        self.query_input = QLineEdit()
        layout.addWidget(self.query_input)

        # Create the send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.start_processing_thread)
        layout.addWidget(self.send_button)

        # Create the loading indicator label
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        movie = QMovie("97930-loading.gif")
        self.loading_label.setMovie(movie)
        movie.start()
        self.loading_label.hide()
        layout.addWidget(self.loading_label)

    def start_processing_thread(self):
        query = self.query_input.text()
        if query:
            self.chat_history_text.append(f"<b>You:</b> {query}\n\n")
            self.query_input.clear()
            self.query_input.setFocus()

            # Disable the send button and show the loading indicator
            self.send_button.setEnabled(False)
            self.loading_label.show()

            # Create and start the worker thread
            self.llm_thread = LLMProcessingThread(query)
            self.llm_thread.response_ready.connect(self.process_llm_response)
            self.llm_thread.start()

    def process_llm_response(self, llm_response):
        result = llm_response['result']
        sources = "\n".join(f"<i>{source.metadata['source']}</i>" for source in llm_response["source_documents"])

        # Append the bot's response to the chat history
        self.chat_history_text.append(f"<b>Bot:</b> {result}\n\n")
        self.chat_history_text.append(f"<b>Sources:</b>\n{sources}\n\n")

        # Clear the loading indicator, enable the send button, and scroll to the bottom of the chat history
        self.loading_label.hide()
        self.send_button.setEnabled(True)
        self.chat_history_text.verticalScrollBar().setValue(self.chat_history_text.verticalScrollBar().maximum())

# Create the application instance
app = QApplication(sys.argv)

# Set the application font
font = QFont("Arial", 14)
app.setFont(font)

# Create the main window
window = ChatBotWindow()
window.show()

# Run the event loop
sys.exit(app.exec())
