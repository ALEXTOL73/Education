
from abc import abstractmethod, abstractproperty

class Data_Transfer(object):
    @abstractmethod
    def prepare(self):
        pass