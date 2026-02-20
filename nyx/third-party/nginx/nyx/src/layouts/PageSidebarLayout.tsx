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

interface PageLayoutProps
{
    title:string;
    actions?:ReactNode;
    searchInput?:ReactNode;
    children:ReactNode;
}



// ==============================
// COMPONENTS
// ==============================

const PageSidebarLayout = ({title, actions, searchInput, children}:PageLayoutProps) =>
{
    return (
        <div className="flex flex-col h-full bg-base-100 border-r border-base-content/10 w-full">

            {/* Header */}
            <div className="p-4">
                <div className={`flex items-center justify-between h-10 ${searchInput ? "mb-4" : ""}`}>
                    <h1 className="text-xl font-bold">{title}</h1>
                    {actions}
                </div>
                {searchInput}
            </div>

            <div className="flex-1 overflow-y-auto p-4">
                {children}
            </div>
        </div>
    );
}


// ==============================
// EXPORTS
// ==============================

export default PageSidebarLayout;