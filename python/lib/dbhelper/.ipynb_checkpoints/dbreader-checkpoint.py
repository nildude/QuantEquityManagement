"""Author: Rajan Subramanian
   Created May 04/2020
   Todo - need to make this class more robust.  
"""
import psycopg2
from configparser import ConfigParser
import pandas as pd
from psycopg2.extras import execute_values, DictCursor
import os


class DbReader:
    """Establishes a sql connection with the PostGres Database
    params:
    None
    Attributes:
    dbconn (conn) connection objevct for psycopg2
    """
    def __init__(self):
        self.conn = None

    def _read_db_config(self, section: str = 'postgresql-dev'):
        """
        Reads the database configuration from config.ini file
        Args:
        filename (string) file where database connection is stored (config.ini)
        section: (string) one of postgressql-dev or postgresql-prod
        Returns:
        config (dict)
        """
        # create the parser
        filename = 'config.ini'

        parser = ConfigParser()
        parser.read(filename)

        # get the section, default to postgressql
        config = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))
        return config

    def connect(self, section: str = 'dev'):
        """Connects to PostGreSql Database
        Args:
        section (string) one of 'dev' or 'prod'
                default to 'dev'

        Returns:
        connection object to database
        """
        if self.conn is None:
            try:
                # read connection parameters
                section = 'postgresql-' + section
                params = self._read_db_config(section=section)
                # connect to the PostgresSql Server
                self.conn = psycopg2.connect(**params)
            except psycopg2.DatabaseError as error:
                print(error)
        else:
            return self.conn

    def fetch(self, query: str, hide: bool = True):
        """Returns the data associated with table
        Args:
        query:  database query parameter
        conn:    connection object (default to None)
                if None, then use dev connection
        hide:   to show status of call

        Returns:
        dictionary of values from database
        """

        try:
            self.connect('dev')
            with self.conn.cursor(cursor_factory=DictCursor) as curr:
                curr.execute(query)
                records = curr.fetchall()
            curr.close()
        except psycopg2.DatabaseError as e:
            print(e)
        else:
            return records

    def push(self, df, conn, hide=False,table_name=None):
        columns = ",".join(list(df))
        #create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) values %s".format(table_name,columns)
        curr = conn.cursor()
        args = list(df.itertuples(index=False, name=None))
        execute_values(curr, insert_stmt, args)
        conn.commit()
        curr.close()

    def drop(self, table_name, conn):
        """removes table given by table_name from dev db"""
        query = f'drop table {table_name};'
        self.execute(query, conn)

    def readTable(self, table_name, limit=10, section='dev'):
        """Reads contents of a table given by table_name
        Args:
        table_name (str): name of the table
        limit (int):      limit observations in table

        Returns:
        Table (DataFrame)
        """
        conn = sql.connect(section)
        query = f"""select * from {table_name}"""
        if limit:
            query += f" " + f"""limit {limit}"""
        df = sql.fetch(query, con=conn)
        conn.close()
        return df



    def delete(self, table_name, section='dev'):
        """deletes all rows given by table_name from dev deb
          table schema is retained
        """
        query = f'delete from {table_name};'
        conn = self.connect(section)
        self.execute(query, dev_conn)
        conn.commit()
        conn.close()

    def execute(self, query, conn):
        curr = conn.cursor()
        curr.execute(query)
        conn.commit()
        curr.close()
        print("Success")
