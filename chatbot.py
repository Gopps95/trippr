from hugchat import hugchat

chatbot = hugchat.ChatBot(cookie_path="cookies.json")
id = chatbot.new_conversation()
chatbot.change_conversation(id)



while True:
 user_input = input('> ')
 if user_input.lower() == '':
  pass
 elif user_input.lower() in ['q', 'quit']:
  break
 elif user_input.lower() in ['c', 'change']:
  print('Choose a conversation to switch to:')
  print(chatbot.get_conversation_list())
 elif user_input.lower() in ['n', 'new']:
  print('Clean slate!')
  id = chatbot.new_conversation()
  chatbot.change_conversation(id)
 else:
  print(chatbot.chat(user_input))