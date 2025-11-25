# ==========================================================================================
# Author: Pablo González García.
# Created: 25/11/2025
# Last edited: 25/11/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import shutil
from typing import Any
from datetime import datetime
from enum import Enum


# ==============================
# CONSTANTES
# ==============================

H1_DECOR:str = '-'


# ==============================
# ENUMS
# ==============================

class ForegroundColor(Enum):
    """
    Representa colores de texto por consola.
    """
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    RESET   = "\033[39m"  # Vuelve al color por defecto

class BackgroundColor(Enum):
    """
    Representa colores de fondo por consola.
    """
    BLACK   = "\033[40m"
    RED     = "\033[41m"
    GREEN   = "\033[42m"
    YELLOW  = "\033[43m"
    BLUE    = "\033[44m"
    MAGENTA = "\033[45m"
    CYAN    = "\033[46m"
    WHITE   = "\033[47m"
    RESET   = "\033[49m"  # Vuelve al color de fondo por defecto


# ==============================
# FUNCIONES
# ==============================

def print_h1(
    text:str,
    spacing_up:int|None = None,
    spacing_down:int|None = None
) -> None:
    """
    Imprime una cabecera h1 por consola.

    Args:
        text (str): Texto de la cabecera.
        spacing_up (int|None): Número de espacios por encima de la cabecera.
        spacing_down (int|None): Número de espacios por debajo de la cabecera.
    """
    # Obtiene el ancho de la terminal.
    width:int = shutil.get_terminal_size().columns

    # Obtiene el largo del título. 
    # Se suman 2 por los espacios al principio y al final.
    text_width:int = len(text) + 2

    # Calcula el padding.
    padding:int = (width - text_width) // 2

    # Calcula el espaciado por encima.
    padding_up:str = f"{"\n"*spacing_up}" if spacing_up is not None else ""
    # Calcula el espaciado por debajo.
    padding_down:str = f"{"\n"*spacing_down}" if spacing_down is not None else ""

    # Imprime el título.
    print(f"{padding_up}{H1_DECOR * padding} {text.upper()} {H1_DECOR * padding}{padding_down}")

def print_attribute(
    key:str,
    value:Any
) -> None:
    """
    Imprime un atributo por consola. Permite representar el valor
    del atributo con color.

    - `key`: `value`
    """
    # Imprime el valor.
    print(f"{key}: {ForegroundColor.MAGENTA.value}{value}{ForegroundColor.RESET.value}")

def print_message_with_label(
    label:str,
    msg:str
) -> None:
    """
    Imprime un mensaje con la forma:

    - dd/mm/yyyy hh:mm:ss [`label`] `msg`
    """
    # Imprime el mensaje.
    print(f"{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} [{label}] {msg}")

def info(
    text:str
) -> None:
    """
    Imprime un mensaje de información con la forma:

    - dd/mm/yyyy hh:mm:ss [INFO] `text`
    """
    # Imprime el mensaje.
    print_message_with_label(
        label=f"{ForegroundColor.CYAN.value}INFO{ForegroundColor.RESET.value}",
        msg=text
    )

def error(
    text:str
) -> None:
    """
    Imprime un mensaje de información con la forma:

    - dd/mm/yyyy hh:mm:ss [ERROR] `text`
    """
    # Imprime el mensaje.
    print_message_with_label(
        label=f"{ForegroundColor.RED.value}ERROR{ForegroundColor.RESET.value}",
        msg=text
    )

def warning(
    text:str
) -> None:
    """
    Imprime un mensaje de información con la forma:

    - dd/mm/yyyy hh:mm:ss [WARNING] `text`
    """
    # Imprime el mensaje.
    print_message_with_label(
        label=F"{ForegroundColor.YELLOW.value}WARNING{ForegroundColor.RESET.value}",
        msg=text
    )