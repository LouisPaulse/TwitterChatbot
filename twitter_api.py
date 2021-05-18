import twitter
from api_keys import api_key, api_secret_key, bearer_Token, access_token, access_token_secret

api = twitter.Api(consumer_key=api_key,
                  consumer_secret=api_secret_key,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret,)

users = api.GetUser(screen_name="louis_paulse")
print(users)


def get_direct_messages():
    users = api.GetDirectMessages(since_id=0, count=200)
    print(users)


def send_direct_message(user):
    pass


def log_message_in_database():
    pass


get_direct_messages()
