// ==========================================================================================
// Author: Pablo González García.
// Created: 19/02/2026
// Last edited: 20/02/2026
// ==========================================================================================


// ==============================
// IMPORTS
// ==============================

// External:
import { create } from "zustand";

// Internal:
import type { Chat, ChatState, Message } from "./types";


// ==============================
// INSTANCES
// ==============================

export const useChatStore = create<ChatState>((set) => ({

    // ---- Properties ---- //

    chats: [],
    selectedChatId: null,
    messages: [],


    // ---- Methods ---- //

    selectChat: (id:string | null) =>
    {
        set({ selectedChatId: id });
    },

    createChat: () =>
    {
        const newChat:Chat = {
            id: crypto.randomUUID(),
            title: "Nueva conversación",
            lastMessage: "",
            updatedAt: new Date().toISOString(),
            isPinned: false
        };

        set((state) => ({
            chats: [newChat, ...state.chats],
            selectedChatId: newChat.id
        }));
    },

    deleteChat: (id:string) =>
    {
        set((state) => ({
            chats: state.chats.filter((chat) => chat.id !== id),
            messages: state.messages.filter((msg) => msg.chatId !== id),
            selectedChatId: state.selectedChatId === id ? null : state.selectedChatId
        }));
    },

    pinChat: (id:string) =>
    {
        set((state) => ({
            chats: state.chats.map((chat) =>
                chat.id === id ? { ...chat, isPinned: !chat.isPinned } : chat
            )
        }));
    },

    renameChat: (id:string, title:string) =>
    {
        set((state) => ({
            chats: state.chats.map((chat) =>
                chat.id === id ? { ...chat, title } : chat
            )
        }));
    },

    sendMessage: (chatId:string, content:string) =>
    {
        const newMessage:Message = {
            id: crypto.randomUUID(),
            chatId,
            content,
            role: "user",
            createdAt: new Date().toISOString()
        };

        set((state) => ({
            messages: [...state.messages, newMessage],
            chats: state.chats.map((chat) =>
                chat.id === chatId
                    ? { ...chat, lastMessage: content, updatedAt: newMessage.createdAt }
                    : chat
            )
        }));
    }
}));