import sqlite3

def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        email TEXT,
        password TEXT,
        father_name TEXT,
        dob TEXT,
        aadhaar TEXT,
        phone TEXT,
        photo TEXT
    )''')
    conn.commit()
    conn.close()

create_users_table()
