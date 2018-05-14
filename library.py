book=[]

class library:
    
    def addBook(self,bookName):
        self.bookName = bookName
        book.append(self.bookName)
        self.printBook()
        
    def printBook(self):
        print (book)
        
    def removeBook(self,bookName):
        self.bookName = bookName
        book.remove(self.bookName)
        self.printBook()

b = library()
while (1):

    n=int(input("Welcome to the Library Add book Press 1 " \
                                    "Remove the book press 2 :"))
    if n == 1:
        booka= input("Add Book Name :")
        b.addBook(booka)
    elif n == 2:
        bookr= input("Remove Book Name :")
        b.removeBook(bookr)
    else :
        print("Enter only 1 or 2 ")
	


