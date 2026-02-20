// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import type { ReactNode } from "react";


// ==============================
// PROPERTIES
// ==============================

interface SidebarSectionProps
{
    children: ReactNode;
    isFirst?: boolean;
}


// ==============================
// COMPONENTS
// ==============================

const SidebarSection = ({ children, isFirst = false }:SidebarSectionProps) =>
{
    return (
        <div className={`py-4 ${isFirst ? "border-y" : "border-b"} border-base-content/10`}>
            {children}
        </div>
    );
};


// ==============================
// EXPORTS
// ==============================

export default SidebarSection;