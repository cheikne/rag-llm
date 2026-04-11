from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5):
        """
        Must return a list of chunks (dictionaries) 
        containing at least 'text' and 'metadata'.
        """
        pass
