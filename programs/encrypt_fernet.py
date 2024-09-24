'''
This is a little program to make an 16-bit AES cipher of your DB password

How to run:
conda install cryptography

python encrypt_fernet.py PASSWORD

It will save two files

pw.txt  the contents of the encrypted PASSWORD
conf/key.txt the string of the key generated

You will then SCP your pw.txt and conf/key.txt files to the server in those locations (DO NOT COMMIT THESE)

Then, when you use python_modules.global_vars in your program, the password will be automatically
'''
from cryptography.fernet import Fernet
###additional function: How to decrypt


if __name__=='__main__':
    import random, string, sys, os, cryptography
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    f = Fernet(key)
    print('Your Key is : {}'.format(key))
    print('This will be saved for your records')
    ###Now save the key
    keyfile = open('key.txt', 'wb')
    keyfile.write(key)
    keyfile.close()
    ###now encrypt the password
    token = f.encrypt(bytes(sys.argv[1].replace("'",''), 'utf-8'))
    tokenfile = open("pw.txt", 'wb')
    tokenfile.write(token)
    tokenfile.close()
    os._exit(0)