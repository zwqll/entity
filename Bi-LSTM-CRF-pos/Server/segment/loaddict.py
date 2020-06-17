mysqlLogin = logIn = {'host': '118.194.132.112',
             'user': 'root',
             'db':'DarkMatter',
             'passwd': 'begin@2015',
             'charset': 'utf8'}
table='EntityDict'

def getdict():
    mysqlHandler = MySQLUtil(**logIn)
    for record in mysqlHandler.select(table=table):
        print(record)