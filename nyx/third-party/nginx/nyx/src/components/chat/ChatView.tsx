
import { ArrowLeft, Send, MessageSquare } from "lucide-react";
import type { Chat } from "../../store/chat/types";

export interface Message
{
    id: string;
    chatId: string;
    text: string;
    sender: "me" | "other";
    time: string;
}

interface ChatViewProps
{
    chat: Chat | null;
    messages: Message[];
    onBack: () => void;
}

const ChatView = ({ chat, messages, onBack }: ChatViewProps) =>
{
    if (!chat)
    {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-base-100 text-base-content/40">
                <MessageSquare className="size-16" />
                <p className="text-lg">Selecciona una conversación</p>
            </div>
        );
    }

    const initials = chat.title
        .split(" ")
        .map((w) => w[0])
        .join("")
        .slice(0, 2)
        .toUpperCase();

    return (
        <div className="flex-1 flex flex-col h-screen bg-base-100">

            {/* Header */}
            <div className="flex items-center gap-3 p-4 border-b border-base-content/10">
                <button className="btn btn-ghost btn-sm btn-square md:hidden" onClick={onBack}>
                    <ArrowLeft className="size-5" />
                </button>
                <div className="size-10 rounded-full bg-primary text-primary-content flex items-center justify-center font-semibold shrink-0">
                    {initials}
                </div>
                <span className="font-medium">{chat.title}</span>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-1">
                {messages.map((msg) => (
                    <div key={msg.id} className={`chat ${msg.sender === "me" ? "chat-end" : "chat-start"}`}>
                        <div className={`chat-bubble ${msg.sender === "me" ? "chat-bubble-primary" : ""}`}>
                            {msg.text}
                        </div>
                        <div className="chat-footer text-xs text-base-content/40 mt-1">
                            {msg.time}
                        </div>
                    </div>
                ))}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-base-content/10">
                <div className="flex items-center gap-2">
                    <input
                        type="text"
                        placeholder="Escribe un mensaje..."
                        className="input input-bordered flex-1"
                        readOnly
                    />
                    <button className="btn btn-primary btn-square">
                        <Send className="size-5" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatView;
