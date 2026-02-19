import { MessageCircle, Settings } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface NavbarProps {
  username: string;
}

const Navbar = ({username}: NavbarProps) =>
{
    const firstLetter = username.charAt(0).toUpperCase();

    const navigate = useNavigate();

    return (
        <div className="flex flex-col h-screen w-20 bg-base-100 border-r border-base-content/10 justify-between
            items-center py-4">
            {/* Top icons */}
            <div className="flex flex-col items-center gap-4">
                <div className="tooltip tooltip-right" data-tip="Chat">
                    <button
                        className="btn btn-ghost btn-circle"
                        onClick={() => navigate("/")}
                    >
                        <MessageCircle size={24}/>
                    </button>
                </div>
            </div>
            {/* Bottom section */}
            <div className="flex flex-col items-center gap-4">
                <div className="tooltip tooltip-right" data-tip="Ajustes">
                    <button
                        className="btn btn-ghost btn-circle"
                        onClick={() => navigate("/settings")}
                    >
                        <Settings />
                    </button>
                </div>
                <div className="tooltip tooltip-right" data-tip="Perfil">
                    <button
                        className="btn btn-ghost btn-circle avatar"
                        onClick={() => navigate("/profile")}
                    >
                        <div className="w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center">
                            {firstLetter}
                        </div>
                    </button>
                </div>
            </div>
        </div>
    )
}

export default Navbar;
