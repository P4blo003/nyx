// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import ChatSidebar from "../components/chat/ChatSidebar";
import ChatPanel from "../components/chat/ChatPanel";
import { useChatStore } from "../store/chat/useChatStore";


// ==============================
// COMPONENTS
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


// ==============================
// EXPORTS
// ==============================

export default HomePage;