import type { LucideIcon } from "lucide-react";
import { FileText, MessageCircle, Settings } from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { memo, useCallback } from "react";


// ==============================
// Types
// ==============================

interface NavbarProps
{
    username: string;
}

interface NavRoute
{
    path: string;
    label: string;
    icon: LucideIcon;
}


// ==============================
// Navigation config
// ==============================

const topRoutes:NavRoute[] = [
    { path: "/", label: "Chat", icon: MessageCircle },
    { path: "/documents", label: "Documentos", icon: FileText }
];

const bottomRoutes:NavRoute[] = [
    { path: "/settings", label: "Ajustes", icon: Settings }
];

const allRoutes:NavRoute[] = [...topRoutes, ...bottomRoutes];


// ==============================
// NavButton
// ==============================

interface NavButtonProps
{
    route: NavRoute;
    isActive: boolean;
    withTooltip: boolean;
    onNavigate: (path:string) => void;
}

const NavButton = memo(({ route, isActive, withTooltip, onNavigate }:NavButtonProps) =>
{
    const { path, label, icon: Icon } = route;

    const handleClick = useCallback(() => onNavigate(path), [onNavigate, path]);

    const className = `btn btn-ghost btn-square rounded-xl transition-all duration-200 hover:bg-base-content/10 ${isActive ? "text-primary scale-110 bg-base-content/10" : ""}`;

    const button = (
        <button className={className} onClick={handleClick}>
            <Icon size={24}/>
        </button>
    );

    if (!withTooltip) return button;

    return (
        <div className="tooltip tooltip-right" data-tip={label}>
            {button}
        </div>
    );
});


// ==============================
// Navbar
// ==============================

const Navbar = ({ username }:NavbarProps) =>
{
    const firstLetter = username.charAt(0).toUpperCase();

    const navigate = useNavigate();
    const location = useLocation();

    const handleNavigate = useCallback((path:string) => navigate(path), [navigate]);

    return (
        <>
            {/* Desktop: vertical sidebar */}
            <div className="hidden md:flex flex-col h-screen w-20 bg-base-300 border-r border-base-content/10
                justify-between items-center py-4">
                <div className="flex flex-col items-center gap-4">
                    {topRoutes.map((route) => (
                        <NavButton
                            key={route.path}
                            route={route}
                            isActive={location.pathname === route.path}
                            withTooltip
                            onNavigate={handleNavigate}
                        />
                    ))}
                </div>
                <div className="flex flex-col items-center gap-4">
                    {bottomRoutes.map((route) => (
                        <NavButton
                            key={route.path}
                            route={route}
                            isActive={location.pathname === route.path}
                            withTooltip
                            onNavigate={handleNavigate}
                        />
                    ))}
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

            {/* Mobile: bottom bar */}
            <div className="flex md:hidden w-full bg-base-300 border-t border-base-content/10
                justify-center items-center px-4 py-2">
                <div className="flex flex-row items-center gap-6">
                    {allRoutes.map((route) => (
                        <NavButton
                            key={route.path}
                            route={route}
                            isActive={location.pathname === route.path}
                            withTooltip={false}
                            onNavigate={handleNavigate}
                        />
                    ))}
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
        </>
    );
};

export default Navbar;
