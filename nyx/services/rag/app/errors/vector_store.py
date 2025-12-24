# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================



# ==============================
# CLASSES
# ==============================

class VectorStoreError(Exception):
    """
    General vector store error.
    """

    # ---- Default ---- #

    def __init__(
        self,
        msg:str,
        code:int
    ) -> None:
        """
        Initialize the error.

        Args:
            msg (str): Exception error.
            code (int): Error code.
        """

        # Exception constructor.
        super().__init__(msg)

        # Initialize the properties.
        self._code:int = code
    

    # ---- Properties ---- #

    @property
    def Code(self) -> int:
        """
        Gets the error code.

        Returns:
            int: Error code.
        """
        return self._code
    
class CollectionExist(VectorStoreError):
    """
    Error when collection already exist.
    """

    # ---- Constants ---- #

    _CODE_:int = 409


    # ---- Default ---- #

    def __init__(
        self,
        msg: str
    ) -> None:
        """
        Initialize the error.

        Args:
            msg (str): Exception error.
        """

        # VectorStoreError constructor.
        super().__init__(
            msg=msg,
            code=self._CODE_
        )

class CollectionNotExist(VectorStoreError):
    """
    Error when collection doesn't exist.
    """

    # ---- Constants ---- #

    _CODE_:int = 404


    # ---- Default ---- #

    def __init__(
        self,
        msg: str
    ) -> None:
        """
        Initialize the error.

        Args:
            msg (str): Exception error.
        """

        # VectorStoreError constructor.
        super().__init__(
            msg=msg,
            code=self._CODE_
        )