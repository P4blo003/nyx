import { memo } from "react";
import type { Message } from "../../store/chat/types";


// ==============================
// Props
// ==============================

interface MessageBubbleProps
{
    message: Message;
}


// ==============================
// Component
// ==============================

const MessageBubble = memo(({ message }:MessageBubbleProps) =>
{
    const isUser = message.role === "user";

    const time = new Date(message.createdAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-center"}`}>
            <div className={`max-w-[75%] px-4 py-2 rounded-2xl ${isUser
                ? "bg-primary text-primary-content rounded-br-sm"
                : "bg-base-200 text-base-content rounded-bl-sm"
            }`}>
                <p className="whitespace-pre-wrap wrap-break-word">{message.content}</p>
                <p className={`text-xs mt-1 ${isUser ? "text-primary-content/60" : "text-base-content/40"}`}>
                    {time}
                </p>
            </div>
        </div>
    );
});

export default MessageBubble;
