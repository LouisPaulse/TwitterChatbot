from database import DatabaseImplementation
from twitter_api import TwitterAPIFunctions

from NLP import nlp_implementation

import time


class Main:
    def __init__(self):
        self.API_CURRENT_RATE = 0
        self.API_RATE_LIMIT = 15  # 15 calls per 15 min on direct message retrieval
        self.twitter_api = TwitterAPIFunctions()
        self.database_obj = DatabaseImplementation(
            database_ini_location="database.ini",
            database_section="postgresql",
        )
        self.nlp_obj = nlp_implementation.NLPImplementation('./NLP/intents.json')

    def update_database(self):
        list_messages = self.twitter_api.list_direct_messages_to_be_stored()

        # First stores users so that the foreign keys can be registered for message_id
        for message in list_messages:
            self.database_obj.store_user_info_database(
                user_id=message["user_id"],
                user_name=message["user_name"],
                user_screen_name=message["user_screen_name"],
                user_location=message["location"],
            )
        for message in list_messages:
            # print(message)
            if message["recipient_id"] != self.twitter_api.My_Message_ID:
                self.database_obj.store_bot_message_info_database(
                    message_id=message["message_id"],
                    user_id=message["user_id"],
                    recipient_user_id=message["recipient_id"],
                    message_text=message["message_data"]["text"],
                    timestamp=message["time_stamp"],
                )
            else:
                self.database_obj.store_user_message_info_database(
                    message_id=message["message_id"],
                    user_id=message["user_id"],
                    message_text=message["message_data"]["text"],
                    message_answered=message["message_answered"],
                    timestamp=message["time_stamp"],
                )

    def nlp_model_go(self, message_text):
        return self.nlp_obj.response(message_text)

    def send_messages_to_unanswered_recipients_texts(self):
        items = self.database_obj.get_unanswered_messages()
        for i in items:
            print(f"Question: {i['message_text']}")
            outcome = self.nlp_model_go(i['message_text'])
            print(f"This is the outcome: {outcome}")

            self.twitter_api.send_direct_message(user_id=i['user_id'], message=outcome)
            self.database_obj.update_user_answered_message(message_id=i['message_id'], answer=True)


if __name__ == '__main__':

    while True:
        main = Main()
        print("Retrieving Messages from twitter and storing them in Database")
        main.update_database()
        print("Retrieval Complete")

        print("Now responding to all messages retrieved")
        main.send_messages_to_unanswered_recipients_texts()

        # Maintains a 1min wait time before requesting new data from twitter (Adheres to API 15min / 15 requests)
        # On direct message retrievals
        time.sleep(60)
