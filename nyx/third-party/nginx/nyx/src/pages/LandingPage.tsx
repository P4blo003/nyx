import { LogIn, Sparkles } from "lucide-react";
import { useAuthStore } from "../store/auth/useAuthStore";


const LandingPage = () =>
{
    const { login } = useAuthStore();

    return (
        <div className="min-h-screen flex flex-col items-center justify-center gap-6 p-8 text-center">
            <div className="size-24 rounded-full bg-primary/10 flex items-center justify-center">
                <Sparkles className="size-12 text-primary" />
            </div>
            <div className="space-y-2">
                <h1 className="text-3xl font-bold">Bienvenido a Nyx</h1>
                <p className="text-base-content/60 max-w-sm">
                    Plataforma de inferencia ML y RAG. Inicia sesión para comenzar.
                </p>
            </div>
            <button
                className="btn btn-primary btn-lg"
                onClick={() => login()}
            >
                <LogIn className="size-5" />
                Iniciar sesión
            </button>
        </div>
    );
}

export default LandingPage;
