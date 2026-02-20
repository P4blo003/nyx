import ChatSidebar from "../components/chat/ChatSidebar";
import ChatPanel from "../components/chat/ChatPanel";
import { useChatStore } from "../store/chat/useChatStore";


// ==============================
// Component
// ==============================

const HomePage = () =>
{
    const selectedChatId = useChatStore((s) => s.selectedChatId);

    return (
        <div className="flex h-full overflow-hidden">

            {/* Sidebar — always visible on md+, hidden on mobile when a chat is selected */}
            <div className={`${selectedChatId ? "hidden" : "w-full"} md:block md:w-80 shrink-0`}>
                <ChatSidebar/>
            </div>

            {/* Chat panel — always visible on md+, hidden on mobile when no chat selected */}
            <div className={`${selectedChatId ? "flex flex-1" : "hidden"} md:flex md:flex-1 min-w-0`}>
                <ChatPanel/>
            </div>
        </div>
    );
};

export default HomePage;
