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
    const chats = useChatStore((s) => s.chats);
    const selectedChatId = useChatStore((s) => s.selectedChatId);
    const selectChat = useChatStore((s) => s.selectChat);
    const createChat = useChatStore((s) => s.createChat);
    const deleteChat = useChatStore((s) => s.deleteChat);
    const pinChat = useChatStore((s) => s.pinChat);
    const renameChat = useChatStore((s) => s.renameChat);

    return (
        <PageSidebarLayout
            title="Chat"
            searchInput={<SearchInput/>}
            actions={
                <div className="tooltip tooltip-bottom" data-tip="Nuevo chat">
                    <button
                        className="btn btn-ghost btn-circle"
                        onClick={createChat}
                    >
                        <MessageSquarePlus size={20}/>
                    </button>
                </div>
            }
        >
            {/* Chat list */}
            <SidebarSection isFirst>
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
                                isSelected={chat.id === selectedChatId}
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
