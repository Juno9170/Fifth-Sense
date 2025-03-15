import socket

s = socket.socket()
print ("Socket successfully created")

port = 12346

s.bind(('0.0.0.0', port))
print ("socket binded to %s" %(port))

s.listen(5)	 
print ("socket is listening")

c, addr = s.accept()
print ('Got connection from', addr )

while True:
    c, addr = s.accept()

    print(c.recv(1024).decode())

    #c.close()

    #break