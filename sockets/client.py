# Import socket module 
import socket             
 
# Create a socket object 
s = socket.socket()         
 
# Define the port on which you want to connect 
port = 12346               
 
# connect to the server on local computer 
s.connect(('192.168.2.96', port)) 

# receive data from the server and decoding to get the string.
print (s.recv(1024).decode("utf-8"))
# close the connection 
s.close()