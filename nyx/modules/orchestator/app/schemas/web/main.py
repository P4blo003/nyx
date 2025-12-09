# ==========================================================================================
# Author: Pablo González García.
# Created: 09/12/2025
# Last edited: 09/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
from typing import Union, Annotated
# External:
from pydantic import Field
# Internal:
from schemas.web.heartbeat import HeartbeatResponse


# ==============================
# TYPES
# ==============================

ApiResponse:Annotated = Annotated[
    Union[
        HeartbeatResponse
    ],
    Field(discriminator='rtype')
]