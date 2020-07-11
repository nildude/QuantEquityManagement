"""
PostGresSQL python Interface for interacting with Dbeaver DB
Author: Rajan Subramanian
Created May 10 2020
Todo - add a copy from and time profiler 
"""
import psycopg2, os, time, pandas as pd
from functools import wraps
from configparser import ConfigParser
from psycopg2.extras import execute_values, DictCursor, execute_batch
from typing import Iterator, List, Tuple, Dict, Any

class DbReader:
    """Establishes a sql connection with the PostGres Database
    params:
    None
    Attributes:
    conn (conn) connection objevct for psycopg2
    """

    def __init__(self):
        self.conn = None

    def _read_db_config(self, section: str = 'postgresql-dev') -> Dict:
        """
        Reads the database configuration from config.ini file
        Args:
        section: one of postgressql-dev or postgresql-prod
        Returns:
        DataBase Configuration
        """
        # create the parser
        filename =  'config.ini'

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
        if self.conn is None or self.conn.closed == True:
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

    def _create_records(self, dictrow: List) -> Tuple:
        """converts data obtained from db into tuple of dictionaries"""
        return tuple({k:v for k,v in record.items()} for record in dictrow)

    def fetch(self, query: str, section: str = 'dev'):
        """Returns the data associated with table
        Args:
        query:  database query parameter
        records:   specify if the rows should be returned as records

        Returns:
        list of DictRows where each item is a dictionary
        """

        try:
            self.connect(section)
            with self.conn.cursor(cursor_factory=DictCursor) as curr:
                curr.execute(query)
                # get column names
                self.col_names = tuple(map(lambda x: x.name, curr.description))
                # fetch the rows
                rows = curr.fetchall()
            self.conn.close()
        except psycopg2.DatabaseError as e:
            print(e)
        else:
            return rows

    def fetchdf(self, query: str, section: str = 'dev'):
        """Returns a pandas dataframe of the db query"""
        return pd.DataFrame(self.fetch(query, section), columns=self.col_names)

    def iterator_from_df(self, datadf: pd.DataFrame) -> Iterator:
        """Convenience function to transform pandas dataframe to 
            Iterator for db push
        """
        yield from iter(datadf.to_dict(orient='rows'))

    def push(self, data: Iterator[Dict[str, Any]], table_name: str, columns: List[str], section: str = 'dev') -> None:
        try:
            self.connect(section)
            with self.conn.cursor() as curr:
                # get the column names
                col_names = ",".join(columns)
                # create an iterator of tuple rows for db insert
                args = (tuple(item.values()) for item in data)
                query = """INSERT INTO {} ({}) values %s""".format(table_name, col_names)
                execute_values(curr, query, args)
            self.conn.commit()
            self.conn.close()
        except psycopg2.DatabaseError as e:
            print(e)
        else:
            return 

    def pushdf(self, datadf: pd.DataFrame, table_name: str, section: str = 'dev') -> None:
        """pushes a pandas dataframe to DataBase"""
        col_names = list(datadf)
        data = self.iterator_from_df(datadf)
        self.push(data, table_name=table_name, columns=col_names, section=section)
        return 

    def copy_from(self, data: Iterator[Dict[str, Any]], table_name: str, columns: List[str], section: str = 'dev') -> None:
        """copies data from csv file and writes to Db"""
        raise NotImplementedError("Will be Implemented Later")
        


    def push1(self, df, conn, hide=False,table_name=None):
        """Deprecated and no longer used"""
        raise DeprecationWarning("function has been deprecated and no longer used")
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

    def readTable(self, table_name: str, limit: int=10, section: str='dev'):
        """Reads contents of a table given by table_name
        Args:
        table_name (str): name of the table
        limit (int):      limit observations in table

        Returns:
        Table (DataFrame)
        """
        query = f"""select * from {table_name}"""
        if limit:
            query += f" " + f"""limit {limit}"""
        return self.fetchdf(query, section=section)



    def delete(self, table_name, section='dev'):
        """deletes all rows given by table_name from dev deb
          table schema is retained
        """
        query = f'delete from {table_name};'
        self.execute(query, section)

    def execute(self, query: str, section: str = 'dev'):
        try:
            self.connect(section)
            with self.conn.cursor() as curr:
                curr.execute(query)
            self.conn.commit()
            self.conn.close()
        except Exception as e:
            print(e)
        else:
            return 
