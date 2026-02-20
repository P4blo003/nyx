import { ArrowLeft } from "lucide-react";
import { useChatStore } from "../../store/chat/useChatStore";
import MessageList from "./MessageList";
import ChatInput from "./ChatInput";
import { useCallback, useMemo } from "react";


// ==============================
// Component
// ==============================

const ChatPanel = () =>
{
    const selectedChatId = useChatStore((s) => s.selectedChatId);
    const selectChat = useChatStore((s) => s.selectChat);
    const messages = useChatStore((s) => s.messages);
    const sendMessage = useChatStore((s) => s.sendMessage);

    const filteredMessages = useMemo(
        () => messages.filter((m) => m.chatId === selectedChatId),
        [messages, selectedChatId]
    );

    const handleGoBack = useCallback(() => selectChat(null), [selectChat]);

    const handleSendMessage = useCallback(
        (content:string) => { if (selectedChatId) sendMessage(selectedChatId, content); },
        [sendMessage, selectedChatId]
    );

    if (!selectedChatId)
    {
        return (
            <div className="flex-1 flex items-center justify-center bg-base-100 text-base-content/40">
                <p className="text-sm">Selecciona una conversación para comenzar</p>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col bg-base-100 min-w-0">

            {/* Mobile back bar */}
            <div className="flex md:hidden items-center h-10 px-2 border-b border-base-content/10">
                <button
                    className="btn btn-ghost btn-square rounded-xl btn-sm"
                    onClick={handleGoBack}
                >
                    <ArrowLeft size={18}/>
                </button>
            </div>

            <MessageList messages={filteredMessages}/>
            <ChatInput onSendMessage={handleSendMessage}/>
        </div>
    );
};

export default ChatPanel;
