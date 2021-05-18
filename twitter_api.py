import tweepy
from api_keys import api_key, api_secret_key, bearer_Token, access_token, access_token_secret

auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def get_user_info(user_id):
    """ retrieves basic user information, to be saved in database"""
    user = api.get_user(user_id=user_id)
    name = user.name
    screen_name = user.screen_name
    location = user.location
    return name, screen_name, location


def list_direct_messages_to_be_stored():
    """ Method retrieves all direct messages using the api and returns a list of objects to be stored in database"""
    messages = api.list_direct_messages()
    information_to_be_stored = []
    for message in messages:
        user_id = message.message_create["sender_id"]
        recipient_id = message.message_create["target"]["recipient_id"]

        user_name, user_screen_name, location = get_user_info(user_id)

        message_text = message.message_create["message_data"]
        time_stamp = message.created_timestamp
        message_answered = False

        to_be_stored = {
            "user_id": user_id,
            "user_name": user_name,
            "user_screen_name": user_screen_name,
            "location": location,
            "message_id": message.id,
            "message_data": message_text,
            "recipient_id": recipient_id,
            "time_stamp": time_stamp,
            "message_answered": message_answered,
        }
        information_to_be_stored.append(to_be_stored)

    return information_to_be_stored


def send_direct_message(user_id, message):
    api.send_direct_message(recipient_id=user_id, text=message)

