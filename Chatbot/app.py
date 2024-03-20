from flask import Flask, render_template, request
from hugchat import hugchat

app = Flask(__name__)

# Initialize the chatbot
chatbot = hugchat.ChatBot(cookie_path="cookies.json")
id = chatbot.new_conversation()
chatbot.change_conversation(id)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    if user_input.lower() == '':
        pass
    elif user_input.lower() in ['q', 'quit']:
        # Handle quit action
        pass
    elif user_input.lower() in ['c', 'change']:
        # Handle change conversation action
        conversation_list = chatbot.get_conversation_list()
        return render_template('change_conversation.html', conversation_list=conversation_list)
    elif user_input.lower() in ['n', 'new']:
        # Handle new conversation action
        id = chatbot.new_conversation()
        chatbot.change_conversation(id)
        response = "Clean slate!"
    else:
        response = chatbot.chat(user_input)

    return response

if __name__ == '__main__':
    app.run(debug=True)
