import psycopg2
from configparser import ConfigParser


def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE users (
            user_id VARCHAR PRIMARY KEY,
            user_name VARCHAR(255) NOT NULL,
            user_screen_name VARCHAR(255) NOT NULL,
            location VARCHAR(255)
        )
        """,
        """ CREATE TABLE user_messages (
                message_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users (user_id),
                message_text VARCHAR NOT NULL,
                answered_text BOOLEAN NOT NULL,
                timestamp VARCHAR NOT NULL
                )
        """,
        """
        CREATE TABLE bot_messages (
                message_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                FOREIGN KEY (user_id)
                    REFERENCES users (user_id),
                recipient_user_id VARCHAR NOT NULL,
                message_text VARCHAR NOT NULL,
                timestamp  VARCHAR NOT NULL
        )
        """)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def store_user_info_database(user_id, user_name, user_screen_name, user_location):
    """ Stores base user information """
    sql = """INSERT INTO users(user_id, user_name, user_screen_name, location)
    VALUES(%s, %s, %s, %s) RETURNING user_id;"""

    print(f"Storing User {user_name} with screen_name {user_screen_name} and id {user_id} into database ")

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (user_id, user_name, user_screen_name, user_location))
        # get the generated id back
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def store_user_message_info_database(message_id, user_id, message_text, message_answered, timestamp):
    """ Stores direct messages retrieved from twitter"""
    sql = """INSERT INTO user_messages(message_id, user_id, message_text, answered_text, timestamp)
    VALUES(%s, %s, %s, %s, %s) RETURNING user_id;"""

    print(f"MessageID {message_id} with UserID {user_id} and text: {message_text}. Answered? {message_answered}")

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (message_id, user_id, message_text, message_answered, timestamp))
        # get the generated id back
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def store_bot_message_info_database(message_id, user_id, recipient_user_id, message_text, timestamp):
    sql = """INSERT INTO bot_messages(message_id, user_id, recipient_user_id, message_text, timestamp)
    VALUES(%s, %s, %s, %s, %s) RETURNING user_id;"""

    print(f"MessageID {message_id} with UserID {user_id} and text: {message_text}.")

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (message_id, user_id, recipient_user_id, message_text, timestamp))
        # get the generated id back
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def update_user_answered_message(message_id, answer=True):
    """ update answer to message based on the message_id """
    sql = """ UPDATE user_messages
                SET answered_text = %s
                WHERE message_id = %s"""
    conn = None
    updated_rows = 0
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the UPDATE  statement
        cur.execute(sql, (answer, message_id))
        # get the number of updated rows
        updated_rows = cur.rowcount
        # Commit the changes to the database
        conn.commit()
        # Close communication with the PostgreSQL database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return updated_rows


def iter_row(cursor, size=10):
    while True:
        rows = cursor.fetchmany(size)
        if not rows:
            break
        for row in rows:
            yield row


def get_unanswered_messages():
    """ query data from the user_messages table and retrieve batch of 10 unanswered messages sorted by earliest time """
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("""
        SELECT user_id, message_id, message_text, timestamp FROM user_messages
         WHERE answered_text=False
         ORDER BY timestamp;
        """)
        print("Number of unanswered messages: ", cur.rowcount)
        items = []
        for row in iter_row(cur, 10):
            item = {'user_id': row[0], 'message_id': row[1], 'message_text': row[2], 'timestamp': row[3]}
            items.append(item)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

    return items


if __name__ == '__main__':
    connect()
    create_tables()
    items = get_unanswered_messages()
    print(items)
