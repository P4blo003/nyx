import type { ReactNode } from "react";


interface PageLayoutProps
{
    title:string;
    searchInput?:ReactNode
    children:ReactNode;
}


const PageSidebarLayout = ({title, searchInput, children}:PageLayoutProps) =>
{
    return (
        <div className="flex flex-col h-screen bg-base-100 border-r border-base-content/10 w-130">

            {/* Header */}
            <div className="p-4 pb-0">
                <h1 className="text-xl font-bold mb-4">{title}</h1>
                    {searchInput}
            </div>

            <div className="flex-1 overflow-y-auto p-4">
                {children}
            </div>
        </div>
    );
}

export default PageSidebarLayout;