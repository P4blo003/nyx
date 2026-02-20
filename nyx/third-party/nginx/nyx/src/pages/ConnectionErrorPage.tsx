// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import { WifiOff } from 'lucide-react';


// ==============================
// COMPONENTS
// ==============================

const ConnectionErrorPage = () =>
{
    return (
        <div className="min-h-screen flex flex-col items-center justify-center gap-6 p-8 text-center">
            <div className="size-24 rounded-full bg-error/10 flex items-center justify-center">
                <WifiOff className="size-12 text-error" />
            </div>
            <div className="space-y-2">
                <h1 className="text-3xl font-bold">No se puede conectar</h1>
                <p className="text-base-content/60 max-w-sm">
                    No hemos podido contactar con el servidor. Comprueba tu conexión o inténtalo más tarde.
                </p>
            </div>
            <button
                className="btn btn-primary btn-lg"
                onClick={() => window.location.reload()}
            >
                Reintentar
            </button>
        </div>
    );
}


// ==============================
// EXPORTS
// ==============================

export default ConnectionErrorPage;