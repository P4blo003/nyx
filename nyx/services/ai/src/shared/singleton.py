# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import TypeVar, Type, Optional


# ==============================
# TYPE
# ==============================

T = TypeVar("T", bound="Singleton")


# ==============================
# IMPORTS
# ==============================

class Singleton:
    """
    Base class for implementing the Singleton design pattern.

    This class ensures that only one instance of a subclass exists
    throughout the application. Subclasses can use `initialize()` to
    create the instance with specific arguments and `get()` to retrieve it.
    """

    # ---- Properties ---- #

    _instance:Optional["Singleton"] = None


    # ---- Methods ---- #

    @classmethod
    def initialize(
        cls:Type[T],
        *args,
        **kargs
    ) -> T:
        """
        Initialize the singleton instance with provided arguments.

        If the instance is already initialized, this method returns the existing
        instance without creating a new one.

        Args:
            *args: Positional arguments to pass to the constructor.
            **kargs: Keyword arguments to pass to the constructor.

        Returns:
            T: The singleton instance of the class.
        """

        # Checks if the instance is already initialized.
        if cls._instance is None: cls._instance = cls(*args, **kargs)
        return cls._instance                                    # type: ignore[return-value]
    
    @classmethod
    def get(cls:Type[T]) -> T:
        """
        Retrieve the singleton instance.

        Raises:
            ValueError: If the singleton has not been initialized yet.

        Returns:
            T: The existing singleton instance.
        """
        
        # Checks if the instance is not initialized.
        if cls._instance is None: raise ValueError(f"{cls.__name__} not initialized. Call initialize() first.")
        return cls._instance                                     # type: ignore[return-value]