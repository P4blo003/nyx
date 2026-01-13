# ==========================================================================================
# Author: Pablo González García.
# Created: 13/01/2026
# Last edited: 13/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import APIRouter, Request
from fastapi import HTTPException
from fastapi import status
# Internal:
from core.tokenizer import TokenizerService
from dto.request import TokenizeRequestDto


# ==============================
# CONSTANTS
# ==============================

# Initialize endpoint.
router:APIRouter = APIRouter(prefix="/tokens")


# ==============================
# FUNCTIONS
# ==============================

@router.post("/")
async def tokenize(
    req:Request,
    tr:TokenizeRequestDto
):
    """
    Tokenize a given query or list of queries.

    Args:
        req (Request): Request's context.
        tr (TokenizeRequestDto): Tokenize request content.
    
    Returns:
    """

    # Try-Except to manage errors.
    try:
        
        # Gets tokenizer.
        tokenizer:TokenizerService = req.app.state.tokenizer

    # If an unexpected error occurs.
    except Exception as ex:

        # Returns Server error.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Unexpected error while tokenizing query."
        )