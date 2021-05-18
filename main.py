from database import (
    store_user_info_database,
    store_user_message_info_database,
    store_bot_message_info_database,
    update_user_answered_message,
    get_unanswered_messages,
)
from twitter_api import send_direct_message, list_direct_messages_to_be_stored

API_CURRENT_RATE = 0
API_RATE_LIMIT = 15  # 15 calls per 15 min on direct message retrieval
MY_MESSAGE_ID = '1389887696986910725'  # My twitter accounts unique message id

### TODO
### TODO: Send to NLP model
### TODO: Receive Output from model


def update_database():
    list_messages = list_direct_messages_to_be_stored()

    # First stores users so that the foreign keys can be registered for message_id
    for message in list_messages:
        store_user_info_database(
            user_id=message["user_id"],
            user_name=message["user_name"],
            user_screen_name=message["user_screen_name"],
            user_location=message["location"],
        )
    for message in list_messages:
        print(message)
        if message["recipient_id"] != MY_MESSAGE_ID:
            store_bot_message_info_database(
                message_id=message["message_id"],
                user_id=message["user_id"],
                recipient_user_id=message["recipient_id"],
                message_text=message["message_data"]["text"],
                timestamp=message["time_stamp"],
            )
        else:
            store_user_message_info_database(
                message_id=message["message_id"],
                user_id=message["user_id"],
                message_text=message["message_data"]["text"],
                message_answered=message["message_answered"],
                timestamp=message["time_stamp"],
            )


def nlp_model_go(message_text):
    outcome = "CHEERS! To the first of many automated messages"
    return outcome


def send_messages_to_unanswered_recipients_texts():
    items = get_unanswered_messages()
    for i in items:
        outcome = nlp_model_go(i['message_text'])

        send_direct_message(user_id=i['user_id'], message=outcome)
        update_user_answered_message(message_id=i['message_id'], answer=True)


if __name__ == '__main__':
    # send_direct_message('788526745364422656', auto)
    # update_answered_text_after_bot_sends_message('1394717946849763339', '788526745364422656', auto)
    send_messages_to_unanswered_recipients_texts()
    print("hello")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
