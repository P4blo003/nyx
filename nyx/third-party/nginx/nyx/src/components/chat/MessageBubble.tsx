// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import { memo } from "react";

// Internal:
import type { Message } from "../../store/chat/types";


// ==============================
// PROPERTIES
// ==============================

interface MessageBubbleProps
{
    message: Message;
}


// ==============================
// COMPONENTS
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


// ==============================
// EXPORTS
// ==============================

export default MessageBubble;