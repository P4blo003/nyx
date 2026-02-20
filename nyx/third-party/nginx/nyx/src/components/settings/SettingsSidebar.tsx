import { LogOut } from "lucide-react";
import PageSidebarLayout from "../../layouts/PageSidebarLayout";
import SidebarSection from "../global/SidebarSection";
import { useAuthStore } from "../../store/auth/useAuthStore";
import SearchInput from "../global/SearchInput";


const SettingsSidebar = () =>
{
    const logout = useAuthStore((s) => s.logout);

    return (
        <PageSidebarLayout
            title="Ajustes"
            searchInput={<SearchInput/>}
        >
            <SidebarSection isFirst>
                <div
                    className="flex items-center gap-3 p-3 rounded-lg cursor-pointer
                    hover:bg-base-300 transition-colors duration-200 text-red-400"
                    onClick={logout}
                >
                    <LogOut size={20}/>
                    <span className="font-medium">Log out</span>
                </div>
            </SidebarSection>
        </PageSidebarLayout>
    );
};

export default SettingsSidebar;
