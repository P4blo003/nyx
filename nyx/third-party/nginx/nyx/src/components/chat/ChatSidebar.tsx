import { MessageSquarePlus } from "lucide-react";
import PageSidebarLayout from "../../layouts/PageSidebarLayout";
import SidebarSection from "../global/SidebarSection";
import ChatListItem from "./ChatListItem";
import { useChatStore } from "../../store/chat/useChatStore";
import SearchInput from "../global/SearchInput";


// ==============================
// Component
// ==============================

const ChatSidebar = () =>
{
    const { chats, selectChat, createChat, deleteChat, pinChat, renameChat } = useChatStore();

    return (
        <PageSidebarLayout
            title="Chat"
            searchInput={<SearchInput/>}
        >
            {/* New chat */}
            <SidebarSection isFirst>
                <button
                    className="btn btn-primary w-full"
                    onClick={createChat}
                >
                    <MessageSquarePlus size={18}/>
                    Nuevo chat
                </button>
            </SidebarSection>

            {/* Chat list */}
            <SidebarSection>
                {chats.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-base-content/40">
                        <p className="text-sm">No hay conversaciones</p>
                    </div>
                ) : (
                    <div className="flex flex-col gap-1">
                        {chats.map((chat) => (
                            <ChatListItem
                                key={chat.id}
                                chat={chat}
                                onSelect={selectChat}
                                onPin={pinChat}
                                onRename={renameChat}
                                onDelete={deleteChat}
                            />
                        ))}
                    </div>
                )}
            </SidebarSection>
        </PageSidebarLayout>
    );
};

export default ChatSidebar;
