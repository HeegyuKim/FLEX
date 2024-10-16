import os
import sqlite3
import pandas as pd



class BaseDatabase:
    def execute(self, db_id: str, query: str, to_pandas: bool = False):
        pass


def load_database(db_engine: str, **kwargs):
    if db_engine == "sqlite":
        return SQLiteDatabase(**kwargs)
    elif db_engine == "mysql":
        return MySQLDatabase(**kwargs)
    elif db_engine == "postgresql":
        return PostgresDatabase(**kwargs)
    else:
        raise ValueError(f"Unsupported database engine: {db_engine}")


class SQLiteDatabase(BaseDatabase):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute(self, db_id: str, query: str, to_pandas: bool = False):

        conn = sqlite3.connect(os.path.join(self.db_path, db_id, f"{db_id}.sqlite"))
        conn.text_factory = lambda b: b.decode(errors = 'ignore')

        with conn:
            if to_pandas:
                result = pd.read_sql_query(query, conn)
            else:
                cursor = conn.cursor()
                cursor.execute(query)
                columns = list(map(lambda x: x[0], cursor.description))
                result = cursor.fetchall()
                return result, columns

        return result
            


class MySQLDatabase(BaseDatabase):
    def __init__(self, host: str, user: str, password: str, db: str):
        import pymysql
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
            cursorclass=pymysql.cursors.DictCursor,
        )
        self.cursor = self.conn.cursor()

    def execute(self, query: str, to_pandas: bool = False):
        if to_pandas:
            return pd.read_sql_query(query, self.conn)
        else:
            self.cursor.execute(query)
            columns = list(map(lambda x: x[0], self.cursor.description))
            result = self.cursor.fetchall()
            print(result, columns)
            return result, columns

    def __del__(self):
        self.conn.close()


class PostgresDatabase(BaseDatabase):
    def __init__(self, host: str, user: str, password: str, db: str):
        import psycopg2
        self.conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            dbname=db,
        )
        self.cursor = self.conn.cursor()

    def execute(self, query: str, to_pandas: bool = False):
        if to_pandas:
            return pd.read_sql_query(query, self.conn)
        else:
            self.cursor.execute(query)
            return self.cursor.fetchall()

    def __del__(self):
        self.conn.close()