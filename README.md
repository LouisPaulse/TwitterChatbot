# Twitter Chatbot
The twitter chatbot interacts with the TwitterAPI to retrieve and send direct messages. Messages are then stored in a relational database to be retrieved later on. The messages not yet replied to by the bot, are processed by a retrieval based nlp model and a return response by the model is then sent again to the respective users twitter direct message feed.

Installation instructions:

### Twitter API
Firstly register a account to access TwittersAPI (https://developer.twitter.com/en/docs/twitter-api). Then follow the instructions here (https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) to create a project app. The provided keys and tokens can then be added to this projects api_keys.py file.


### Twitter direct messages
Once logged in to your twitter account, proceed to the Messages tab and then settings. Enable the "Allow message requests from everyone" and "Show read receipts". 


### Database
PostgreSQL is used in this project. Details on how to install can be found here (https://www.postgresql.org/download/). A database should then be created along with a user and password. Those details can then be added to the projects database.ini file. 

### Requirements
Project dependencies can be installed using the following commands:
(pip install -r requirements.txt)
(pip install -U pip setuptools wheel)
(pip install -U spacy)
(python -m spacy download en_core_web_sm)


