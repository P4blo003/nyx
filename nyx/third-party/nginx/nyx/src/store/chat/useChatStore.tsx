import { create } from "zustand";
import type { Chat, ChatState } from "./types";


export const useChatStore = create<ChatState>((set) => ({

    // ---- Properties ---- //

    chats: [],
    selectedChatId: null,


    // ---- Methods ---- //

    selectChat: (id:string) =>
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
    }
}));
