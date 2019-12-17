# Python3 program to illustrate store
# efficiently using pickle module
# Module translates an in-memory Python object
# into a serialized byte stream—a string of
# bytes that can be written to any file-like object.

import pickle

dogs_dict = { 'Ozzy': 3, 'Filou': 8, 'Luna': 5, 'Skippy': 10, 'Barco': 12, 'Balou': 9, 'Laika': 16 }
import bz2
import pickle

sfile = bz2.BZ2File('smallerfile', 'w')
pickle.dump(dogs_dict, sfile)
# def storeData():
#     # initializing data to be stored in db
#     Omkar = {'key': 'Omkar', 'name': 'Omkar Pathak',
#              'age': 21, 'pay': 40000}
#     Jagdish = {'key': 'Jagdish', 'name': 'Jagdish Pathak',
#                'age': 50, 'pay': 50000}
#
#     # database
#     db = {}
#     db['Omkar'] = Omkar
#     db['Jagdish'] = Jagdish
#
#     # Its important to use binary mode
#     dbfile = open('examplePickle', 'ab')
#
#     # source, destination
#     pickle.dump(db, dbfile)
#     dbfile.close()
#
#
# def loadData():
#     # for reading also binary mode is important
#     dbfile = open('examplePickle', 'rb')
#     db = pickle.load(dbfile)
#     for keys in db:
#         print(keys, '=>', db[keys])
#     dbfile.close()
#
#
# if __name__ == '__main__':
#     storeData()
#     loadData()
#
# # class MyClass:
# #     def method(self):
# #         return 'instance method called', self
# #
# #     @classmethod
# #     def classmethod(cls):
# #         return 'class method called', cls
# #
# #     @staticmethod
# #     def staticmethod():
# #         return 'static method called'
# # obj = MyClass()
# print(obj.method())