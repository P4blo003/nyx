// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// Standard:
import { useEffect, useRef } from "react";

// Internal:
import type { Message } from "../../store/chat/types";
import MessageBubble from "./MessageBubble";


// ==============================
// PROPERTIES
// ==============================

interface MessageListProps
{
    messages: Message[];
}


// ==============================
// COMPONENTS
// ==============================

const MessageList = ({ messages }:MessageListProps) =>
{
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() =>
    {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    if (messages.length === 0)
    {
        return (
            <div className="flex-1 flex items-center justify-center text-base-content/40">
                <p className="text-sm">Envía un mensaje para comenzar</p>
            </div>
        );
    }

    return (
        <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-3" style={{ scrollbarGutter: "stable" }}>
            {messages.map((message) => (
                <MessageBubble key={message.id} message={message}/>
            ))}
            <div ref={bottomRef}/>
        </div>
    );
};


// ==============================
// EXPORTS
// ==============================

export default MessageList;