# ==========================================================================================
# Author: Pablo González García.
# Created: 01/12/2025
# Last edited: 01/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Estándar:
import os
import logging
import logging.config
from typing import Dict, Any


# ==============================
# CONSTANTES
# ==============================

# Configuración por defecto del logger.
DEFAULT_CONFIG:Dict[str, Any] = {
    'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '[%(asctime)s] %(levelname)s %(name)s - %(message)s'
            },

        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'error': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'standard',
                'filename': os.path.join("logs", 'errors.log'),
                'maxBytes': 10485760,
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'training': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'filename': os.path.join("logs", 'trainings.log'),
                'maxBytes': 10485760,
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'error'],
            'propagate': False
        },
        'loggers': {
            'app': {
                'level': 'DEBUG',
                'handlers': ['console', 'error'],
                'propagate': False
            },
            'training': {
                'level': 'INFO',
                'handlers': ['console', 'error', 'training'],
                'propagate': False
            }
        }
}


# ==============================
# CONFIGURACIÓN
# ==============================

# Inicializa el logging.
logging.config.dictConfig(DEFAULT_CONFIG)


# ==============================
# FUNCIONES
# ==============================

def get_logger(name:str):
    """
    Crea un logger en función del nombre dado.

    Args:
        name (str): Nombre del logger.
    
    Returns:
        logging.Logger: Logger configurado.
    """
    # Retorna el logger.
    return logging.getLogger(name=name)