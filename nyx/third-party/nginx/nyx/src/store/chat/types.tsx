// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// INTERFACES
// ==============================

export interface Chat
{
    id: string;
    title: string;
    lastMessage: string;
    updatedAt: string;
    isPinned: boolean;
}

export interface Message
{
    id: string;
    chatId: string;
    content: string;
    role: "user" | "assistant";
    createdAt: string;
}

export interface ChatState
{
    // ---- Properties ---- //

    chats: Chat[];
    selectedChatId: string | null;
    messages: Message[];


    // ---- Methods ---- //

    selectChat: (id: string | null) => void;
    createChat: () => void;
    deleteChat: (id: string) => void;
    pinChat: (id: string) => void;
    renameChat: (id: string, title: string) => void;
    sendMessage: (chatId: string, content: string) => void;
}
