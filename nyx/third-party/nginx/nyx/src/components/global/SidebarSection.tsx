import type { ReactNode } from "react";


// ==============================
// Props
// ==============================

interface SidebarSectionProps
{
    children: ReactNode;
    isFirst?: boolean;
}


// ==============================
// Component
// ==============================

const SidebarSection = ({ children, isFirst = false }:SidebarSectionProps) =>
{
    return (
        <div className={`py-4 ${isFirst ? "border-y" : "border-b"} border-base-content/10`}>
            {children}
        </div>
    );
};

export default SidebarSection;
