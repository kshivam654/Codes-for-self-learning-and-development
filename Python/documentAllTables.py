import psycopg2

#establishing the connection
conn = psycopg2.connect(
   database="postgres", user='postgres', password='password', host='127.0.0.1', port= '5432'
)
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Executing an MYSQL function using the execute() method to get all the table names
cursor.execute("select version()")

print("Connection established to: ",data)

#Closing the connection
conn.close()