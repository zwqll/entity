import MySQLdb  

class MySQLUtil:  
   
    def __init__(self, **login):
        '''
        The constructor of MysqlOperator class.

        Creates an instance of MysqlOperator and starts connecting
        to the MySQL Database.

        :Args:
         - **login - Keyword arguments used in pymysql.Connect()
           function in order to connect to database and authenticate
           clients (including host, user, password, database, port,
           charset and so on).
        '''
        self.__connector = MySQLdb.connect(**login)
        
    def close(self):
        '''
        Closes connection to MySQL Database.

        This method has to be called after you have done with
        MysqlOperator.
        '''
        self.__connector.close()
    
    def select(self, column=(), condition='', table='test'):
        '''
        Select records or columns from MySQL Database specified by
        condition, default from test table.(condition is equivalent
        to the where clause in SQL select syntax)

        :Args:
         - *column - list of column names in variable argument.
           Used to specify which columns or whole record to return.
         - condition - str. Used to specify conditions to match
           records in MySQL Database, default in test table.
         - table - str. Named keyword argument to specify which
           table to execute.

        :Returns:
         - results - If select success, returns a DictCursor for
         navigating select results.If not, returns None.
        '''
        # prepare sql statement:
        if column is ():
            keys = '*'
        else:
            keys = ', '.join(column)
        if condition is not '':
            condition = " where "+condition
        sql = "select "+keys+" from "+table+condition
        # select transaction:
        try:
            cur = self.__connector.cursor()
            cur.execute(sql)
            results = cur.fetchall()
            self.__connector.commit()
            # print('>> MySQL Select operation success.')
        except MySQLdb.Error as e:
            print('>> MySQL Select operation fail:', e)
            self.__connector.rollback()
            results = None
            # add exception logging function later:
            
        return results
             
        
# test:
if __name__ == '__main__':
    logIn = {'host': '118.194.132.112',
             'user': 'root',
             'db':'DarkMatter',
             'passwd': 'begin@2015',
             'charset': 'utf8'}   
    mysqlHandler = MySQLUtil(**logIn)
    datas = mysqlHandler.select(table='User')
    print datas
    for record in mysqlHandler.select(table='User'):
        print(record)
        
     